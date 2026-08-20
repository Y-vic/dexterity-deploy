using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;
using Microsoft.Win32;
using StereoKit;

namespace PNDQuestTeleop;

[SupportedOSPlatform("windows")]
internal static class Program
{
    private const string ClientVersion = "v18";
    private static readonly TimeSpan HandLogInterval = TimeSpan.FromMilliseconds(200);
    private static readonly string RosHost = GetSetting(
        "PND_QUEST_ROS_HOST",
        "10.10.20.127");
    private static readonly string ZedHost = GetSetting(
        "PND_QUEST_ZED_HOST",
        "10.10.20.126");
    private static readonly int ZedPort = GetPortSetting(
        "PND_QUEST_ZED_PORT",
        5602);
    private static readonly object LogLock = new();
    private static readonly string PrimaryLogPath = Path.Combine(
        AppContext.BaseDirectory,
        "quest-teleop.log");
    private static readonly string FallbackLogPath = Path.Combine(
        Path.GetTempPath(),
        "PNDQuestTeleop.log");
    private static string activeLogPath = PrimaryLogPath;

    private static QuestTrackingClient? trackingClient;
    private static ZedVideoReceiver? videoReceiver;
    private static TextStyle coordinateTextStyle;
    private static readonly ControllerPoseTracker LeftControllerPose = new("Left");
    private static readonly ControllerPoseTracker RightControllerPose = new("Right");
    private static Vec3? leftPositionOrigin;
    private static Vec3? rightPositionOrigin;
    private static bool calibratePressed;
    private static long lastHandLogTimestamp;

    [STAThread]
    private static int Main(string[] args)
    {
        try
        {
            return Run(args);
        }
        catch (Exception exception)
        {
            WriteLog($"Unhandled startup error: {exception}");
            ShowError(
                "PND Quest Teleoperation failed to start.\n\n" +
                $"{exception.Message}\n\nLog: {activeLogPath}");
            return 3;
        }
    }

    private static int Run(string[] args)
    {
        int sessionId = Process.GetCurrentProcess().SessionId;
        Environment.SetEnvironmentVariable(
            "DISABLE_XR_APILAYER_MANUS_handtracking",
            "1");
        WriteLog(
            $"Starting PND Quest Teleoperation {ClientVersion}; " +
            $"PID={Environment.ProcessId}; " +
            $"Session={sessionId}; User={Environment.UserName}; " +
            $"BaseDirectory={AppContext.BaseDirectory}");
        WriteLog($"Active OpenXR runtime: {GetActiveOpenXrRuntime()}");
        WriteLog($"ROS host: {RosHost}; ZED stream: {ZedHost}:{ZedPort}");
        WriteLog("MANUS OpenXR API layer disabled for this process");

        if (args.Contains("--diagnose", StringComparer.OrdinalIgnoreCase))
        {
            WriteLog("Diagnostics completed without starting OpenXR");
            Console.WriteLine($"Diagnostics log: {activeLogPath}");
            return 0;
        }

        if (args.Contains("--video-diagnose", StringComparer.OrdinalIgnoreCase))
        {
            using var receiver = new ZedVideoReceiver(
                ZedHost,
                ZedPort,
                1280,
                720,
                WriteLog,
                verboseLogging: true);
            receiver.Start();
            DateTimeOffset deadline = DateTimeOffset.UtcNow.AddSeconds(15);
            while (DateTimeOffset.UtcNow < deadline)
            {
                receiver.StepDiagnostics();
                Thread.Sleep(100);
            }
            receiver.LogDiagnosticsSummary();
            bool receiving = receiver.State == "Receiving";
            WriteLog($"Video diagnostics result: {receiver.State}");
            return receiving ? 0 : 6;
        }

        if (sessionId == 0)
        {
            WriteLog("Startup blocked: Meta OpenXR is unavailable in Windows Session 0");
            ShowError(
                "Meta OpenXR cannot run from SSH or a Windows service session.\n\n" +
                "Start this application from the PND Quest Teleop shortcut on " +
                "the Windows desktop.\n\n" +
                $"Log: {activeLogPath}");
            return 5;
        }

        Process[] oldInstances = Process.GetProcessesByName("PNDQuestTeleop")
            .Where(process => process.Id != Environment.ProcessId)
            .ToArray();
        if (oldInstances.Length > 0)
        {
            string processIds = string.Join(", ", oldInstances.Select(process => process.Id));
            WriteLog($"Startup blocked by existing PNDQuestTeleop process: {processIds}");
            ShowError(
                "Another PND Quest Teleoperation process is already running.\n\n" +
                $"PID: {processIds}\n\n" +
                "Close it in Task Manager, then start PND Quest Teleop again.\n\n" +
                $"Log: {activeLogPath}");
            return 4;
        }

        Log.Subscribe((level, message) => WriteLog($"StereoKit {level}: {message}"));
        var settings = new SKSettings
        {
            appName = "PND Quest Teleoperation",
            mode = AppMode.XR,
            blendPreference = DisplayBlend.Opaque,
            noFlatscreenFallback = true,
            disableDesktopInputWindow = true,
            origin = OriginMode.Floor,
            standbyMode = StandbyMode.None,
            logFilter = LogLevel.Diagnostic,
        };

        WriteLog("Initializing StereoKit OpenXR");
        bool initialized = SK.Initialize(settings);
        WriteLog($"StereoKit OpenXR initialization result: {initialized}");
        if (!initialized)
        {
            ShowError(
                "OpenXR initialization failed.\n\n" +
                "Confirm Quest Link is active inside the headset and close any " +
                "older PNDQuestTeleop process before retrying.\n\n" +
                $"Log: {activeLogPath}");
            return 1;
        }

        try
        {
            Renderer.EnableSky = false;
            Renderer.ClearColor = new Color(0.035f, 0.04f, 0.045f, 1.0f);
            Input.HandVisible(Handed.Max, false);
            Input.HandSolid(Handed.Max, false);
            coordinateTextStyle = Text.MakeStyle(
                Default.Font,
                0.012f,
                new Color(0.15f, 0.92f, 0.42f, 1.0f));
            trackingClient = new QuestTrackingClient(
                new Uri($"https://{RosHost}/webvr/runtime-config.json"),
                new Uri($"wss://{RosHost}/vrwebsocket"),
                new Uri($"https://{RosHost}"),
                WriteLog);
            trackingClient.Start();
            videoReceiver = new ZedVideoReceiver(
                ZedHost,
                ZedPort,
                1280,
                720,
                WriteLog);
            videoReceiver.Start();
            SK.Run(Step, Shutdown);
            return 0;
        }
        catch (Exception exception)
        {
            WriteLog($"Fatal error: {exception}");
            return 2;
        }
        finally
        {
            videoReceiver?.Dispose();
            videoReceiver = null;
            trackingClient?.Dispose();
            trackingClient = null;
            SK.Shutdown();
        }
    }

    private static void Step()
    {
        Controller left = Input.Controller(Handed.Left);
        Controller right = Input.Controller(Handed.Right);
        Pose? leftPose = LeftControllerPose.Update(left);
        Pose? rightPose = RightControllerPose.Update(right);

        LogHandTracking(left, right, leftPose, rightPose);

        trackingClient?.Publish(
            Input.Head,
            leftPose,
            rightPose,
            left,
            right);

        videoReceiver?.Step(Input.Head);
        bool pressed = right.IsX1Pressed;
        if (pressed && !calibratePressed && leftPose is not null && rightPose is not null)
        {
            leftPositionOrigin = leftPose.Value.position;
            rightPositionOrigin = rightPose.Value.position;
            WriteLog(
                "A button pressed: local display origin reset and retarget " +
                "arms-forward calibration requested; " +
                $"left={FormatVector(leftPositionOrigin.Value)}, " +
                $"right={FormatVector(rightPositionOrigin.Value)}");
        }
        calibratePressed = pressed;

        DrawTrackingAxes(leftPose, leftPositionOrigin);
        DrawTrackingAxes(rightPose, rightPositionOrigin);
    }

    private static void DrawTrackingAxes(Pose? pose, Vec3? positionOrigin)
    {
        if (pose is null)
        {
            return;
        }

        const float axisLength = 0.07f;
        const float axisThickness = 0.0025f;
        const float arrowLength = 0.016f;
        const float arrowWidth = 0.006f;
        Color32 green = new Color32(38, 235, 106, 255);
        Matrix transform = pose.Value.ToMatrix();
        Vec3 origin = pose.Value.position;
        DrawArrow(
            transform,
            origin,
            new Vec3(axisLength, 0, 0),
            new Vec3(axisLength - arrowLength, arrowWidth, 0),
            new Vec3(axisLength - arrowLength, -arrowWidth, 0),
            green,
            axisThickness);
        DrawArrow(
            transform,
            origin,
            new Vec3(0, axisLength, 0),
            new Vec3(arrowWidth, axisLength - arrowLength, 0),
            new Vec3(-arrowWidth, axisLength - arrowLength, 0),
            green,
            axisThickness);
        DrawArrow(
            transform,
            origin,
            new Vec3(0, 0, axisLength),
            new Vec3(arrowWidth, 0, axisLength - arrowLength),
            new Vec3(-arrowWidth, 0, axisLength - arrowLength),
            green,
            axisThickness);

        Vec3 position = positionOrigin is null
            ? pose.Value.position
            : pose.Value.position - positionOrigin.Value;
        Text.Add(
            $"x {position.x:F3}\ny {position.y:F3}\nz {position.z:F3}",
            Matrix.TRS(
                origin,
                Quat.LookAt(origin, Input.Head.position),
                Vec3.One),
            coordinateTextStyle,
            TextAlign.TopLeft,
            TextAlign.TopLeft);
    }

    private static string FormatVector(Vec3 value) =>
        $"({value.x:F3}, {value.y:F3}, {value.z:F3})";

    private static void LogHandTracking(
        Controller left,
        Controller right,
        Pose? leftPose,
        Pose? rightPose)
    {
        long now = Stopwatch.GetTimestamp();
        if (lastHandLogTimestamp != 0
            && Stopwatch.GetElapsedTime(lastHandLogTimestamp, now) < HandLogInterval)
        {
            return;
        }

        lastHandLogTimestamp = now;
        WriteLog(
            $"Hands: Left[{FormatTracking(left, leftPose)}]; " +
            $"Right[{FormatTracking(right, rightPose)}]");
    }

    private static string FormatTracking(Controller controller, Pose? pose)
    {
        string position = pose is null ? "unavailable" : FormatVector(pose.Value.position);
        return
            $"tracked={controller.IsTracked}, " +
            $"position={controller.trackedPos}, rotation={controller.trackedRot}, " +
            $"xyz={position}";
    }

    private static string GetSetting(string name, string defaultValue)
    {
        string? value = Environment.GetEnvironmentVariable(name);
        return string.IsNullOrWhiteSpace(value) ? defaultValue : value.Trim();
    }

    private static int GetPortSetting(string name, int defaultValue)
    {
        string? value = Environment.GetEnvironmentVariable(name);
        return int.TryParse(value, out int port) && port is > 0 and <= 65535
            ? port
            : defaultValue;
    }

    private sealed class ControllerPoseTracker
    {
        private readonly string name;
        private Pose cachedPose = Pose.Identity;
        private bool hasPosition;
        private bool hasRotation;
        private string lastTrackingState = "";

        public ControllerPoseTracker(string name)
        {
            this.name = name;
        }

        public Pose? Update(Controller controller)
        {
            bool positionAvailable =
                controller.trackedPos != TrackState.Lost;
            bool rotationAvailable =
                controller.trackedRot != TrackState.Lost;

            if (positionAvailable)
            {
                cachedPose.position = controller.pose.position;
                hasPosition = true;
            }
            if (rotationAvailable)
            {
                cachedPose.orientation = controller.pose.orientation;
                hasRotation = true;
            }

            string trackingState =
                $"isTracked={controller.IsTracked}, " +
                $"position={controller.trackedPos}, rotation={controller.trackedRot}";
            if (!string.Equals(
                    trackingState,
                    lastTrackingState,
                    StringComparison.Ordinal))
            {
                WriteLog($"{name} controller tracking changed: {trackingState}");
                lastTrackingState = trackingState;
            }

            return positionAvailable && hasPosition && hasRotation
                ? cachedPose
                : null;
        }
    }

    private static void DrawArrow(
        Matrix transform,
        Vec3 origin,
        Vec3 localTip,
        Vec3 localArrowA,
        Vec3 localArrowB,
        Color32 color,
        float thickness)
    {
        Vec3 tip = transform.Transform(localTip);
        Lines.Add(origin, tip, color, thickness);
        Lines.Add(tip, transform.Transform(localArrowA), color, thickness);
        Lines.Add(tip, transform.Transform(localArrowB), color, thickness);
    }

    private static void Shutdown()
    {
        videoReceiver?.Dispose();
        videoReceiver = null;
        trackingClient?.Dispose();
        trackingClient = null;
    }

    private static void WriteLog(string message)
    {
        string line = $"{DateTimeOffset.Now:O} {message}";
        lock (LogLock)
        {
            Console.WriteLine(line);

            try
            {
                File.AppendAllText(activeLogPath, line + Environment.NewLine);
            }
            catch (Exception primaryException) when (activeLogPath != FallbackLogPath)
            {
                activeLogPath = FallbackLogPath;
                try
                {
                    File.AppendAllText(
                        activeLogPath,
                        $"{line}{Environment.NewLine}" +
                        $"Primary log failed: {primaryException.Message}{Environment.NewLine}");
                }
                catch (Exception fallbackException)
                {
                    Console.Error.WriteLine($"Logging failed: {fallbackException.Message}");
                }
            }
        }
    }

    private static string GetActiveOpenXrRuntime()
    {
        const string openXrKey = @"SOFTWARE\Khronos\OpenXR\1";
        using RegistryKey? key = Registry.LocalMachine.OpenSubKey(openXrKey);
        return key?.GetValue("ActiveRuntime") as string ?? "Not configured";
    }

    private static void ShowError(string message)
    {
        MessageBox(IntPtr.Zero, message, "PND Quest Teleoperation", 0x10);
    }

    [DllImport("user32.dll", CharSet = CharSet.Unicode, EntryPoint = "MessageBoxW")]
    private static extern int MessageBox(
        IntPtr window,
        string text,
        string caption,
        uint type);
}
