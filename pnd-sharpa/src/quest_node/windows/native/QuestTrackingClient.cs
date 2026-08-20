using System.Diagnostics;
using System.Net.Http.Json;
using System.Net.WebSockets;
using System.Text;
using System.Text.Json;
using System.Threading.Channels;
using StereoKit;

namespace PNDQuestTeleop;

internal sealed class QuestTrackingClient : IDisposable
{
    private readonly Uri configUri;
    private readonly Uri socketUri;
    private readonly Uri origin;
    private readonly Action<string> log;
    private readonly CancellationTokenSource cancellation = new();
    private static readonly TimeSpan SendTimeout = TimeSpan.FromMilliseconds(250);
    private static readonly TimeSpan DiagnosticsInterval = TimeSpan.FromSeconds(2);
    private readonly Channel<TrackingFrame> outgoing = Channel.CreateBounded<TrackingFrame>(
        new BoundedChannelOptions(1)
        {
            FullMode = BoundedChannelFullMode.DropOldest,
            SingleReader = true,
            SingleWriter = true,
        });

    private Task? worker;
    private string connectionState = "Starting";
    private string calibrationState = "Not calibrated";
    private long producedFrames;
    private long droppedFrames;
    private long sentFrames;
    private long lastDequeuedSequence;
    private long lastDiagnosticsTimestamp;

    public QuestTrackingClient(
        Uri configUri,
        Uri socketUri,
        Uri origin,
        Action<string> log)
    {
        this.configUri = configUri;
        this.socketUri = socketUri;
        this.origin = origin;
        this.log = log;
    }

    public string ConnectionState => Volatile.Read(ref connectionState);
    public string CalibrationState => Volatile.Read(ref calibrationState);

    public void Start()
    {
        worker = Task.Run(() => RunAsync(cancellation.Token));
    }

    public void Publish(
        Pose head,
        Pose? leftPose,
        Pose? rightPose,
        Controller left,
        Controller right)
    {
        long sequence = Interlocked.Increment(ref producedFrames);
        object payload = new
        {
            sequence,
            timestamp = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds(),
            monotonicTimestampMs = Stopwatch.GetTimestamp() * 1000.0 / Stopwatch.Frequency,
            Head = PosePayload(head),
            LeftHand = PosePayload(leftPose, "left", left),
            RightHand = PosePayload(rightPose, "right", right),
            Calibration = new
            {
                pressed = right.IsX1Pressed,
            },
            Joy = new
            {
                axes = new[]
                {
                    left.stick.x,
                    left.stick.y,
                    left.trigger,
                    left.grip,
                    right.stick.x,
                    right.stick.y,
                    right.trigger,
                    right.grip,
                },
                buttons = new[]
                {
                    ButtonPair(left.IsStickClicked),
                    ButtonPair(left.IsX1Pressed),
                    ButtonPair(left.IsX2Pressed),
                    ButtonPair(right.IsStickClicked),
                    ButtonPair(right.IsX1Pressed),
                    ButtonPair(right.IsX2Pressed),
                },
            },
        };
        if (!outgoing.Writer.TryWrite(
                new TrackingFrame(sequence, JsonSerializer.Serialize(payload))))
        {
            Interlocked.Increment(ref droppedFrames);
        }
    }

    private async Task RunAsync(CancellationToken token)
    {
        while (!token.IsCancellationRequested)
        {
            try
            {
                Volatile.Write(ref connectionState, "Loading configuration");
                string accessToken = await LoadAccessTokenAsync(token);
                await RunSocketAsync(accessToken, token);
            }
            catch (OperationCanceledException) when (token.IsCancellationRequested)
            {
                break;
            }
            catch (Exception exception)
            {
                Volatile.Write(ref connectionState, "Disconnected");
                Volatile.Write(ref calibrationState, "Not calibrated");
                log($"Tracking connection failed: {exception.Message}");
                try
                {
                    await Task.Delay(1000, token);
                }
                catch (OperationCanceledException)
                {
                    break;
                }
            }
        }
    }

    private async Task<string> LoadAccessTokenAsync(CancellationToken token)
    {
        using var handler = new HttpClientHandler
        {
            ServerCertificateCustomValidationCallback = ValidateServerCertificate,
        };
        using var client = new HttpClient(handler) { Timeout = TimeSpan.FromSeconds(3) };
        RuntimeConfig? config = await client.GetFromJsonAsync<RuntimeConfig>(configUri, token);
        if (string.IsNullOrWhiteSpace(config?.AccessToken))
        {
            throw new InvalidOperationException("Server did not provide an access token");
        }
        return config.AccessToken;
    }

    private async Task RunSocketAsync(string accessToken, CancellationToken token)
    {
        using var socket = new ClientWebSocket();
        socket.Options.RemoteCertificateValidationCallback = ValidateServerCertificate;
        socket.Options.SetRequestHeader("Origin", origin.GetLeftPart(UriPartial.Authority));
        socket.Options.KeepAliveInterval = TimeSpan.FromSeconds(5);

        Volatile.Write(ref connectionState, "Connecting");
        await socket.ConnectAsync(socketUri, token);
        await SendTextWithTimeoutAsync(
            socket,
            JsonSerializer.Serialize(new { type = "auth", token = accessToken }),
            token);

        string authentication = await ReceiveTextAsync(socket, token);
        using (JsonDocument document = JsonDocument.Parse(authentication))
        {
            if (!document.RootElement.TryGetProperty("type", out JsonElement type)
                || type.GetString() != "auth_ok")
            {
                throw new InvalidOperationException($"Tracking authentication rejected: {authentication}");
            }
        }

        Volatile.Write(ref connectionState, "Connected");
        log("Tracking WebSocket connected");
        using var linked = CancellationTokenSource.CreateLinkedTokenSource(token);
        Task receiver = ReceiveLoopAsync(socket, linked.Token);

        try
        {
            await foreach (TrackingFrame frame in outgoing.Reader.ReadAllAsync(linked.Token))
            {
                long previousSequence = Interlocked.Exchange(
                    ref lastDequeuedSequence,
                    frame.Sequence);
                if (previousSequence == 0 && frame.Sequence > 1)
                {
                    Interlocked.Add(ref droppedFrames, frame.Sequence - 1);
                }
                else if (previousSequence > 0 && frame.Sequence > previousSequence + 1)
                {
                    Interlocked.Add(
                        ref droppedFrames,
                        frame.Sequence - previousSequence - 1);
                }
                if (receiver.IsCompleted)
                {
                    await receiver;
                }
                long sendStarted = Stopwatch.GetTimestamp();
                try
                {
                    await SendTextWithTimeoutAsync(
                        socket,
                        frame.Payload,
                        linked.Token);
                }
                catch
                {
                    Interlocked.Increment(ref droppedFrames);
                    throw;
                }
                Interlocked.Increment(ref sentFrames);
                LogTransportDiagnostics(
                    Stopwatch.GetElapsedTime(sendStarted).TotalMilliseconds);
            }
        }
        finally
        {
            linked.Cancel();
            try
            {
                await receiver;
            }
            catch (OperationCanceledException)
            {
            }
            Volatile.Write(ref connectionState, "Disconnected");
            Volatile.Write(ref calibrationState, "Not calibrated");
        }
    }

    private async Task ReceiveLoopAsync(ClientWebSocket socket, CancellationToken token)
    {
        while (socket.State == WebSocketState.Open && !token.IsCancellationRequested)
        {
            string message = await ReceiveTextAsync(socket, token);
            using JsonDocument document = JsonDocument.Parse(message);
            JsonElement root = document.RootElement;
            if (!root.TryGetProperty("type", out JsonElement type)
                || type.GetString() != "calibration")
            {
                continue;
            }

            string state = root.TryGetProperty("state", out JsonElement stateElement)
                ? stateElement.GetString() ?? "unknown"
                : "unknown";
            Volatile.Write(
                ref calibrationState,
                state switch
                {
                    "calibrated" => "Calibrated",
                    "already_calibrated" => "Calibrated",
                    "rejected" => "Rejected",
                    "reset" => "Not calibrated",
                    _ => state,
                });
            log($"Calibration event: {message}");
        }
        throw new WebSocketException("Tracking socket closed");
    }

    private static async Task SendTextAsync(
        ClientWebSocket socket,
        string message,
        CancellationToken token)
    {
        byte[] data = Encoding.UTF8.GetBytes(message);
        await socket.SendAsync(
            new ArraySegment<byte>(data),
            WebSocketMessageType.Text,
            true,
            token);
    }

    private static async Task SendTextWithTimeoutAsync(
        ClientWebSocket socket,
        string message,
        CancellationToken token)
    {
        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(token);
        timeout.CancelAfter(SendTimeout);
        try
        {
            await SendTextAsync(socket, message, timeout.Token);
        }
        catch (OperationCanceledException) when (!token.IsCancellationRequested)
        {
            throw new TimeoutException(
                $"tracking WebSocket send exceeded {SendTimeout.TotalMilliseconds:F0} ms");
        }
    }

    private void LogTransportDiagnostics(double sendDurationMs)
    {
        long now = Stopwatch.GetTimestamp();
        long previous = Volatile.Read(ref lastDiagnosticsTimestamp);
        if (previous != 0
            && Stopwatch.GetElapsedTime(previous, now) < DiagnosticsInterval)
        {
            return;
        }
        if (Interlocked.CompareExchange(ref lastDiagnosticsTimestamp, now, previous) != previous)
        {
            return;
        }
        log(
            $"Tracking transport: produced={Volatile.Read(ref producedFrames)}, " +
            $"dropped={Volatile.Read(ref droppedFrames)}, " +
            $"sent={Volatile.Read(ref sentFrames)}, " +
            $"last_sent_sequence={Volatile.Read(ref lastDequeuedSequence)}, " +
            $"send_ms={sendDurationMs:F1}");
    }

    private static async Task<string> ReceiveTextAsync(
        ClientWebSocket socket,
        CancellationToken token)
    {
        byte[] buffer = new byte[4096];
        using var message = new MemoryStream();
        WebSocketReceiveResult result;
        do
        {
            result = await socket.ReceiveAsync(new ArraySegment<byte>(buffer), token);
            if (result.MessageType == WebSocketMessageType.Close)
            {
                throw new WebSocketException("Tracking server closed the connection");
            }
            message.Write(buffer, 0, result.Count);
        }
        while (!result.EndOfMessage);
        return Encoding.UTF8.GetString(message.ToArray());
    }

    private static object PosePayload(
        Pose? pose,
        string? hand = null,
        Controller? controller = null) => new
    {
        position = new
        {
            x = (pose ?? Pose.Identity).position.x,
            y = (pose ?? Pose.Identity).position.y,
            z = (pose ?? Pose.Identity).position.z,
        },
        quaternion = new
        {
            x = (pose ?? Pose.Identity).orientation.x,
            y = (pose ?? Pose.Identity).orientation.y,
            z = (pose ?? Pose.Identity).orientation.z,
            w = (pose ?? Pose.Identity).orientation.w,
        },
        hand,
        tracking = TrackingPayload(controller),
    };

    private static object? TrackingPayload(Controller? controller) =>
        controller is null
            ? null
            : new
            {
                connected = controller.IsTracked
                    || controller.trackedPos != TrackState.Lost
                    || controller.trackedRot != TrackState.Lost,
                position = controller.trackedPos.ToString(),
                rotation = controller.trackedRot.ToString(),
            };

    private static int[] ButtonPair(bool pressed) =>
        pressed ? new[] { 1, 1 } : new[] { 0, 0 };

    private static bool ValidateServerCertificate(
        HttpRequestMessage _,
        System.Security.Cryptography.X509Certificates.X509Certificate2? __,
        System.Security.Cryptography.X509Certificates.X509Chain? ___,
        System.Net.Security.SslPolicyErrors ____)
    {
        return true;
    }

    private static bool ValidateServerCertificate(
        object _,
        System.Security.Cryptography.X509Certificates.X509Certificate? __,
        System.Security.Cryptography.X509Certificates.X509Chain? ___,
        System.Net.Security.SslPolicyErrors ____)
    {
        return true;
    }

    public void Dispose()
    {
        if (cancellation.IsCancellationRequested)
        {
            return;
        }
        cancellation.Cancel();
        outgoing.Writer.TryComplete();
        try
        {
            worker?.Wait(TimeSpan.FromSeconds(2));
        }
        catch
        {
        }
        cancellation.Dispose();
    }

    private sealed class RuntimeConfig
    {
        public string AccessToken { get; init; } = "";
    }

    private sealed record TrackingFrame(long Sequence, string Payload);
}
