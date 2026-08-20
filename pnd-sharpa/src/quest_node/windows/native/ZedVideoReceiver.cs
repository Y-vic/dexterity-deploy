using System.Diagnostics;
using System.Runtime.InteropServices;
using LibVLCSharp.Shared;
using StereoKit;

namespace PNDQuestTeleop;

internal sealed class ZedVideoReceiver : IDisposable
{
    private const int DefaultNetworkCacheMilliseconds = 100;
    private readonly object frameLock = new();
    private readonly object decoderLock = new();
    private readonly object reconnectLock = new();
    private readonly Action<string> log;
    private readonly string host;
    private readonly int port;
    private readonly int networkCacheMilliseconds;
    private readonly uint width;
    private readonly uint height;
    private readonly int frameBytes;
    private IntPtr decoderBuffer;
    private IntPtr readyBuffer;
    private IntPtr uploadBuffer;
    private long readySequence;
    private long uploadedSequence;
    private long decodedFrameCount;
    private long uploadedFrameCount;
    private long overwrittenFrameCount;
    private long uploadDurationTicks;
    private long maxUploadDurationTicks;
    private long firstDecodedTimestamp;
    private long lastDecodedTimestamp;
    private long decoderStartedTimestamp;
    private long nextDiagnosticsTimestamp;
    private long diagnosticsTimestamp;
    private long diagnosticsDecodedFrameCount;
    private long diagnosticsUploadedFrameCount;
    private long diagnosticsOverwrittenFrameCount;
    private long diagnosticsUploadDurationTicks;
    private LibVLC? libVlc;
    private Media? media;
    private MediaPlayer? player;
    private Tex? texture;
    private Material? material;
    private Pose screenPose;
    private string state = "Waiting for video";
    private bool firstFrameLogged;
    private int reconnectRequested;
    private int suppressStoppedReconnect;
    private long decoderGeneration;
    private long nextReconnectUtcTicks;
    private bool disposed;

    public ZedVideoReceiver(
        string host,
        int port,
        uint width,
        uint height,
        Action<string> log,
        bool verboseLogging = false)
    {
        this.host = host;
        this.port = port;
        this.width = width;
        this.height = height;
        this.log = log;
        VerboseLogging = verboseLogging;
        networkCacheMilliseconds = GetEnvironmentInteger(
            "PND_QUEST_NETWORK_CACHE_MS",
            DefaultNetworkCacheMilliseconds,
            20,
            2000);
        frameBytes = checked((int)(width * height * 4));
    }

    private bool VerboseLogging { get; }

    public string State => Volatile.Read(ref state);

    public void Start()
    {
        try
        {
            log("ZED video: initializing LibVLC core");
            Core.Initialize();
            log("ZED video: LibVLC core initialized");
            decoderBuffer = Marshal.AllocHGlobal(frameBytes);
            readyBuffer = Marshal.AllocHGlobal(frameBytes);
            uploadBuffer = Marshal.AllocHGlobal(frameBytes);
            diagnosticsTimestamp = Stopwatch.GetTimestamp();
            nextDiagnosticsTimestamp = diagnosticsTimestamp +
                (long)(Stopwatch.Frequency * 5.0);
            log($"ZED video: allocated three {frameBytes} byte native frame buffers");
            log("ZED video: creating LibVLC instance");
            libVlc = new LibVLC(
                "--no-audio",
                "--aout=dummy",
                "--prefetch-buffer-size=256",
                "--prefetch-read-size=262144",
                $"--network-caching={networkCacheMilliseconds}",
                "--clock-jitter=0",
                "--clock-synchro=0");
            if (VerboseLogging)
            {
                libVlc.Log += (_, eventArgs) =>
                    log($"LibVLC {eventArgs.Level}: {eventArgs.Message}");
            }
            if (!RebuildDecoder())
            {
                RequestReconnect("Initial decoder start failed");
            }
        }
        catch (Exception exception)
        {
            SetState("Unavailable");
            log($"ZED video disabled: {exception}");
            ReleaseDecoder();
        }
    }

    public void Step(Pose head)
    {
        Vec3 screenPosition = head.position +
            (head.orientation * Vec3.Forward) * 1.45f;
        screenPose = new Pose(
            screenPosition,
            Quat.LookAt(screenPosition, head.position));
        CheckFrameWatchdog();
        RetryConnectionIfNeeded();
        if (TryTakeFrame(out IntPtr frame))
        {
            EnsureTexture();
            long uploadStarted = Stopwatch.GetTimestamp();
            texture!.SetColors((int)width, (int)height, frame);
            long uploadTicks = Stopwatch.GetTimestamp() - uploadStarted;
            Interlocked.Add(ref uploadDurationTicks, uploadTicks);
            UpdateMaximum(ref maxUploadDurationTicks, uploadTicks);
        }
        LogFrameRatesIfDue();

        if (material is not null)
        {
            const float screenWidth = 1.42f;
            float screenHeight = screenWidth * height / width;
            Mesh.Quad.Draw(
                material,
                Matrix.TRS(
                    screenPose.position,
                    screenPose.orientation,
                    new Vec3(screenWidth, screenHeight, 1.0f)));
        }
    }

    public void StepDiagnostics()
    {
        CheckFrameWatchdog();
        RetryConnectionIfNeeded();
        LogFrameRatesIfDue();
    }

    public void LogDiagnosticsSummary()
    {
        long decoded = Interlocked.Read(ref decodedFrameCount);
        long first = Interlocked.Read(ref firstDecodedTimestamp);
        long last = Interlocked.Read(ref lastDecodedTimestamp);
        double decodedFps = decoded > 1 && last > first
            ? (decoded - 1) * Stopwatch.Frequency / (double)(last - first)
            : 0.0;
        log(
            $"ZED video diagnostics summary: decoded={decoded}, " +
            $"decoded-rate={decodedFps:F1} fps, state={State}");
    }

    private void RequestReconnect(string newState)
    {
        RequestReconnect(newState, 1.0);
    }

    private void RequestReconnect(string newState, double delaySeconds)
    {
        if (Volatile.Read(ref disposed))
        {
            return;
        }
        SetState(newState);
        bool scheduled = false;
        lock (reconnectLock)
        {
            if (reconnectRequested == 0)
            {
                nextReconnectUtcTicks =
                    DateTimeOffset.UtcNow.AddSeconds(delaySeconds).UtcTicks;
                reconnectRequested = 1;
                scheduled = true;
            }
        }
        if (scheduled)
        {
            log($"ZED video: {newState}; decoder rebuild scheduled");
        }
    }

    private void RequestReconnect(
        string newState,
        long generation,
        bool stoppedEvent = false)
    {
        if (generation != Interlocked.Read(ref decoderGeneration) ||
            (stoppedEvent && Volatile.Read(ref suppressStoppedReconnect) != 0))
        {
            return;
        }
        RequestReconnect(newState);
    }

    private void RetryConnectionIfNeeded()
    {
        lock (reconnectLock)
        {
            if (reconnectRequested == 0 ||
                DateTimeOffset.UtcNow.UtcTicks < nextReconnectUtcTicks)
            {
                return;
            }
            reconnectRequested = 0;
        }
        SetState("Rebuilding decoder");
        log("ZED video: rebuilding media player and input");
        if (!RebuildDecoder())
        {
            RequestReconnect("Decoder rebuild failed");
        }
    }

    private bool RebuildDecoder()
    {
        lock (decoderLock)
        {
            if (disposed || libVlc is null)
            {
                return false;
            }

            long generation = Interlocked.Increment(ref decoderGeneration);
            Interlocked.Exchange(ref suppressStoppedReconnect, 1);
            try
            {
                ReleasePlayback();
                lock (frameLock)
                {
                    uploadedSequence = readySequence;
                }
                log($"ZED video: creating media player generation {generation}");
                bool hardwareDecoding = !string.Equals(
                    Environment.GetEnvironmentVariable("PND_QUEST_DISABLE_HW_DECODE"),
                    "1",
                    StringComparison.OrdinalIgnoreCase);
                var newPlayer = new MediaPlayer(libVlc)
                {
                    EnableHardwareDecoding = hardwareDecoding,
                };
                player = newPlayer;
                newPlayer.SetVideoFormat("RV32", width, height, width * 4);
                newPlayer.SetVideoCallbacks(LockVideo, null, DisplayVideo);
                newPlayer.EncounteredError += (_, _) =>
                    RequestReconnect("Decoder error", generation);
                newPlayer.Playing += (_, _) =>
                {
                    if (generation != Interlocked.Read(ref decoderGeneration))
                    {
                        return;
                    }
                    SetState("Decoder started; waiting for frames");
                    log(
                        $"ZED video: media player generation {generation} " +
                        "started; waiting for first frame");
                };
                newPlayer.Stopped += (_, _) =>
                    RequestReconnect("Video input stopped", generation, true);

                string mediaUrl = $"tcp://{host}:{port}";
                var newMedia = new Media(libVlc, mediaUrl, FromType.FromLocation);
                media = newMedia;
                newMedia.AddOption(
                    $":network-caching={networkCacheMilliseconds}");
                newMedia.AddOption(":demux=ts");
                newMedia.AddOption(":input-repeat=-1");
                Interlocked.Exchange(
                    ref decoderStartedTimestamp,
                    Stopwatch.GetTimestamp());
                SetState("Connecting");
                log(
                    $"ZED video: connecting generation {generation} to {mediaUrl}; " +
                    $"hardware decoding " +
                    $"{(hardwareDecoding ? "enabled" : "disabled by environment")}; " +
                    $"network cache={networkCacheMilliseconds} ms; " +
                    "late-frame dropping enabled");
                if (!newPlayer.Play(newMedia))
                {
                    ReleasePlayback();
                    return false;
                }
                return true;
            }
            catch (Exception exception)
            {
                log($"ZED video: decoder rebuild error: {exception}");
                ReleasePlayback();
                return false;
            }
            finally
            {
                Interlocked.Exchange(ref suppressStoppedReconnect, 0);
            }
        }
    }

    private void CheckFrameWatchdog()
    {
        if (Volatile.Read(ref reconnectRequested) != 0 || player is null)
        {
            return;
        }

        long now = Stopwatch.GetTimestamp();
        long lastFrame = Interlocked.Read(ref lastDecodedTimestamp);
        long activity = Math.Max(lastFrame, Interlocked.Read(ref decoderStartedTimestamp));
        if (activity != 0 &&
            now - activity >= (long)(Stopwatch.Frequency * 2.5))
        {
            RequestReconnect("No decoded video frame for 2.5 seconds", 0.0);
        }
    }

    private void EnsureTexture()
    {
        if (texture is not null)
        {
            return;
        }
        texture = new Tex(TexType.ImageNomips | TexType.Dynamic, TexFormat.Bgra32);
        material = Default.MaterialUnlit.Copy();
        material[MatParamName.DiffuseTex] = texture;
    }

    private IntPtr LockVideo(IntPtr opaque, IntPtr planes)
    {
        IntPtr frame;
        lock (frameLock)
        {
            frame = decoderBuffer;
        }
        Marshal.WriteIntPtr(planes, frame);
        return frame;
    }

    private void DisplayVideo(IntPtr opaque, IntPtr picture)
    {
        lock (frameLock)
        {
            if (readySequence != uploadedSequence)
            {
                Interlocked.Increment(ref overwrittenFrameCount);
            }
            (decoderBuffer, readyBuffer) = (readyBuffer, decoderBuffer);
            readySequence++;
        }
        Interlocked.Increment(ref decodedFrameCount);
        long now = Stopwatch.GetTimestamp();
        Interlocked.CompareExchange(ref firstDecodedTimestamp, now, 0);
        Interlocked.Exchange(ref lastDecodedTimestamp, now);
        if (!firstFrameLogged)
        {
            firstFrameLogged = true;
            log($"ZED video: received first decoded {width}x{height} frame");
        }
        SetState("Receiving");
    }

    private bool TryTakeFrame(out IntPtr frame)
    {
        lock (frameLock)
        {
            if (readySequence == uploadedSequence)
            {
                frame = IntPtr.Zero;
                return false;
            }
            (readyBuffer, uploadBuffer) = (uploadBuffer, readyBuffer);
            uploadedSequence = readySequence;
            frame = uploadBuffer;
        }
        Interlocked.Increment(ref uploadedFrameCount);
        return true;
    }

    private void LogFrameRatesIfDue()
    {
        long now = Stopwatch.GetTimestamp();
        if (now < nextDiagnosticsTimestamp)
        {
            return;
        }

        long decoded = Interlocked.Read(ref decodedFrameCount);
        long uploaded = Interlocked.Read(ref uploadedFrameCount);
        long overwritten = Interlocked.Read(ref overwrittenFrameCount);
        long uploadTicks = Interlocked.Read(ref uploadDurationTicks);
        long maxUploadTicks = Interlocked.Exchange(ref maxUploadDurationTicks, 0);
        double elapsedSeconds =
            (now - diagnosticsTimestamp) / (double)Stopwatch.Frequency;
        long lastFrameTimestamp = Interlocked.Read(ref lastDecodedTimestamp);
        string frameAge = lastFrameTimestamp == 0
            ? "never"
            : $"{(now - lastFrameTimestamp) * 1000.0 / Stopwatch.Frequency:F0} ms";
        long uploadedDelta = uploaded - diagnosticsUploadedFrameCount;
        double averageUploadMilliseconds = uploadedDelta == 0
            ? 0.0
            : (uploadTicks - diagnosticsUploadDurationTicks) * 1000.0 /
                (Stopwatch.Frequency * uploadedDelta);
        log(
            $"ZED video rate: decoded=" +
            $"{(decoded - diagnosticsDecodedFrameCount) / elapsedSeconds:F1} fps, " +
            $"uploaded={uploadedDelta / elapsedSeconds:F1} fps, " +
            $"overwritten={overwritten - diagnosticsOverwrittenFrameCount}, " +
            $"upload-copy={averageUploadMilliseconds:F2} ms avg/" +
            $"{maxUploadTicks * 1000.0 / Stopwatch.Frequency:F2} ms max, " +
            $"last-frame-age={frameAge}");
        diagnosticsTimestamp = now;
        diagnosticsDecodedFrameCount = decoded;
        diagnosticsUploadedFrameCount = uploaded;
        diagnosticsOverwrittenFrameCount = overwritten;
        diagnosticsUploadDurationTicks = uploadTicks;
        nextDiagnosticsTimestamp = now + (long)(Stopwatch.Frequency * 5.0);
    }

    private static int GetEnvironmentInteger(
        string name,
        int defaultValue,
        int minimum,
        int maximum)
    {
        string? value = Environment.GetEnvironmentVariable(name);
        return int.TryParse(value, out int parsed) && parsed >= minimum && parsed <= maximum
            ? parsed
            : defaultValue;
    }

    private static void UpdateMaximum(ref long target, long value)
    {
        long current = Volatile.Read(ref target);
        while (value > current)
        {
            long observed = Interlocked.CompareExchange(ref target, value, current);
            if (observed == current)
            {
                return;
            }
            current = observed;
        }
    }

    private void SetState(string value)
    {
        Volatile.Write(ref state, value);
    }

    public void Dispose()
    {
        if (disposed)
        {
            return;
        }
        disposed = true;
        ReleaseDecoder();
        material = null;
        texture = null;
    }

    private void ReleaseDecoder()
    {
        lock (decoderLock)
        {
            Interlocked.Increment(ref decoderGeneration);
            Interlocked.Exchange(ref suppressStoppedReconnect, 1);
            try
            {
                ReleasePlayback();
                libVlc?.Dispose();
                libVlc = null;
            }
            finally
            {
                Interlocked.Exchange(ref suppressStoppedReconnect, 0);
            }
        }
        if (decoderBuffer != IntPtr.Zero)
        {
            Marshal.FreeHGlobal(decoderBuffer);
            decoderBuffer = IntPtr.Zero;
        }
        if (readyBuffer != IntPtr.Zero)
        {
            Marshal.FreeHGlobal(readyBuffer);
            readyBuffer = IntPtr.Zero;
        }
        if (uploadBuffer != IntPtr.Zero)
        {
            Marshal.FreeHGlobal(uploadBuffer);
            uploadBuffer = IntPtr.Zero;
        }
    }

    private void ReleasePlayback()
    {
        MediaPlayer? oldPlayer = player;
        Media? oldMedia = media;
        player = null;
        media = null;
        oldPlayer?.Stop();
        oldMedia?.Dispose();
        oldPlayer?.Dispose();
    }
}
