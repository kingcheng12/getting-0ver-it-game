using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

public static class RLSceneSetup {
    const string MainScenePath = "Assets/Scenes/MainScene.unity";
    const string WaypointRootName = "RL Waypoints";
    const string FirstWaypointName = "Waypoint 1 - First Pillar Exit";
    const string SecondWaypointName = "Waypoint 2 - Second Pillar";

    [MenuItem("RL/Configure Main Scene")]
    public static void ConfigureMainScene() {
        Scene scene = EditorSceneManager.OpenScene(
            MainScenePath, OpenSceneMode.Single);

        GameObject playerObject = GameObject.Find("Player");
        if (playerObject == null) {
            throw new MissingReferenceException(
                "MainScene must contain a GameObject named Player.");
        }

        PlayerControl playerControl =
            playerObject.GetComponent<PlayerControl>();
        if (playerControl == null) {
            throw new MissingComponentException(
                "Player must contain PlayerControl.");
        }

        BehaviorParameters behavior =
            GetOrAddComponent<BehaviorParameters>(playerObject);
        GettingOverItAgent agent =
            GetOrAddComponent<GettingOverItAgent>(playerObject);
        DecisionRequester requester =
            GetOrAddComponent<DecisionRequester>(playerObject);

        Rigidbody2D body = playerControl.body.GetComponent<Rigidbody2D>();
        Rigidbody2D hammer =
            playerControl.hammerHead.GetComponent<Rigidbody2D>();
        Camera camera = Camera.main;

        if (body == null || hammer == null || camera == null) {
            throw new MissingReferenceException(
                "MainScene requires body and hammer Rigidbody2D components " +
                "and a camera tagged MainCamera.");
        }
        RLWaypoint[] waypoints = ConfigureWaypoints(body.position);

        agent.Configure(playerControl, body, hammer, camera, waypoints);
        agent.MaxStep = 0;

        behavior.BehaviorName = "GettingOverIt";
        behavior.BehaviorType = BehaviorType.Default;
        behavior.BrainParameters.VectorObservationSize =
            GettingOverItAgent.ObservationSize;
        behavior.BrainParameters.NumStackedVectorObservations = 1;
        behavior.BrainParameters.ActionSpec =
            ActionSpec.MakeContinuous(
                GettingOverItAgent.ContinuousActionSize);

        requester.DecisionPeriod = 4;
        requester.TakeActionsBetweenDecisions = true;

        EditorUtility.SetDirty(playerControl);
        EditorUtility.SetDirty(agent);
        EditorUtility.SetDirty(behavior);
        EditorUtility.SetDirty(requester);
        EditorSceneManager.MarkSceneDirty(scene);
        EditorSceneManager.SaveScene(scene);

        Debug.Log(
            "Configured MainScene for Gymnasium communication: " +
            "20 observations, 2 continuous actions, decision period 4, " +
            "and two waypoint markers. Drag the waypoint markers to the " +
            "desired body locations in the Scene view before training.");
    }

    [MenuItem("RL/Configure Main Scene", true)]
    static bool CanConfigureMainScene() {
        return !EditorApplication.isPlayingOrWillChangePlaymode;
    }

    [MenuItem("RL/Validate Main Scene")]
    public static void ValidateMainScene() {
        Scene scene = EditorSceneManager.OpenScene(
            MainScenePath, OpenSceneMode.Single);
        GameObject playerObject = GameObject.Find("Player");

        bool valid = playerObject != null &&
                     playerObject.GetComponent<PlayerControl>() != null &&
                     playerObject.GetComponent<GettingOverItAgent>() != null &&
                     playerObject.GetComponent<BehaviorParameters>() != null &&
                     playerObject.GetComponent<DecisionRequester>() != null &&
                     FindWaypoints().Length == 2;

        if (!valid) {
            throw new UnityEditor.Build.BuildFailedException(
                "MainScene RL setup is incomplete. Run " +
                "RL > Configure Main Scene.");
        }

        BehaviorParameters behavior =
            playerObject.GetComponent<BehaviorParameters>();
        DecisionRequester requester =
            playerObject.GetComponent<DecisionRequester>();

        valid =
            behavior.BehaviorName == "GettingOverIt" &&
            behavior.BrainParameters.VectorObservationSize ==
                GettingOverItAgent.ObservationSize &&
            behavior.BrainParameters.ActionSpec.NumContinuousActions ==
                GettingOverItAgent.ContinuousActionSize &&
            requester.DecisionPeriod == 4 &&
            requester.TakeActionsBetweenDecisions;

        if (!valid) {
            throw new UnityEditor.Build.BuildFailedException(
                "MainScene RL component settings are invalid. Run " +
                "RL > Configure Main Scene again.");
        }

        Debug.Log("MainScene RL setup is valid: " + scene.path);
    }

    [MenuItem("RL/Validate Main Scene", true)]
    static bool CanValidateMainScene() {
        return !EditorApplication.isPlayingOrWillChangePlaymode;
    }

    static T GetOrAddComponent<T>(GameObject gameObject)
        where T : Component {
        T component = gameObject.GetComponent<T>();
        return component != null
            ? component
            : Undo.AddComponent<T>(gameObject);
    }

    static RLWaypoint[] ConfigureWaypoints(Vector2 bodyPosition) {
        GameObject root = GameObject.Find(WaypointRootName);
        if (root == null) {
            root = new GameObject(WaypointRootName);
            Undo.RegisterCreatedObjectUndo(root, "Create RL waypoints");
        }

        RLWaypoint first = GetOrCreateWaypoint(
            root.transform,
            FirstWaypointName,
            bodyPosition + new Vector2(3.0f, 1.5f));
        RLWaypoint second = GetOrCreateWaypoint(
            root.transform,
            SecondWaypointName,
            bodyPosition + new Vector2(6.0f, 3.0f));
        return new[] { first, second };
    }

    static RLWaypoint GetOrCreateWaypoint(
        Transform parent,
        string waypointName,
        Vector2 defaultPosition
    ) {
        Transform existing = parent.Find(waypointName);
        GameObject waypointObject;
        if (existing == null) {
            waypointObject = new GameObject(waypointName);
            Undo.RegisterCreatedObjectUndo(
                waypointObject, "Create RL waypoint");
            waypointObject.transform.SetParent(parent, true);
            waypointObject.transform.position = defaultPosition;
        } else {
            waypointObject = existing.gameObject;
        }
        return GetOrAddComponent<RLWaypoint>(waypointObject);
    }

    static RLWaypoint[] FindWaypoints() {
        GameObject root = GameObject.Find(WaypointRootName);
        if (root == null) {
            return new RLWaypoint[0];
        }
        RLWaypoint first = root.transform
            .Find(FirstWaypointName)?.GetComponent<RLWaypoint>();
        RLWaypoint second = root.transform
            .Find(SecondWaypointName)?.GetComponent<RLWaypoint>();
        if (first == null || second == null) {
            return new RLWaypoint[0];
        }
        return new[] { first, second };
    }
}
