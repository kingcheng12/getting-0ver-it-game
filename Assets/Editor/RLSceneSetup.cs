using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

public static class RLSceneSetup {
    const string MainScenePath = "Assets/Scenes/MainScene.unity";

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

        agent.Configure(playerControl, body, hammer, camera);
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
            "20 observations, 2 continuous actions, decision period 4.");
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
                     playerObject.GetComponent<DecisionRequester>() != null;

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

    static T GetOrAddComponent<T>(GameObject gameObject)
        where T : Component {
        T component = gameObject.GetComponent<T>();
        return component != null
            ? component
            : Undo.AddComponent<T>(gameObject);
    }
}
