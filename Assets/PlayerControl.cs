using UnityEngine;

public class PlayerControl : MonoBehaviour {
    const int ContactBufferSize = 8;

    public Transform hammerHead;
    public Transform body;
    public float maxRange = 2.0f;

    readonly Collider2D[] contactResults = new Collider2D[ContactBufferSize];

    Rigidbody2D hammerRigidbody;
    Rigidbody2D bodyRigidbody;
    Collider2D hammerCollider;
    Collider2D bodyCollider;
    ContactFilter2D terrainContactFilter;
    Vector2 externalCommand;
    bool externalControlEnabled;

    public bool HammerTouchingTerrain { get; private set; }
    public Vector2 LastAppliedCommand { get; private set; }

    void Awake() {
        CacheComponents();
        InitializeContactFilter();
        Physics2D.IgnoreCollision(hammerCollider, bodyCollider);
    }

    void FixedUpdate() {
        Vector2 command =
            externalControlEnabled ? externalCommand : GetMouseCommand();
        ApplyNormalizedCommand(command);
    }

    void CacheComponents() {
        if (hammerHead == null || body == null) {
            throw new MissingReferenceException(
                "PlayerControl requires hammerHead and body transforms.");
        }

        hammerRigidbody = hammerHead.GetComponent<Rigidbody2D>();
        bodyRigidbody = body.GetComponent<Rigidbody2D>();
        hammerCollider = hammerHead.GetComponent<Collider2D>();
        bodyCollider = body.GetComponent<Collider2D>();

        if (hammerRigidbody == null || bodyRigidbody == null ||
            hammerCollider == null || bodyCollider == null) {
            throw new MissingComponentException(
                "PlayerControl requires Rigidbody2D and Collider2D components " +
                "on both hammerHead and body.");
        }
    }

    void InitializeContactFilter() {
        terrainContactFilter = new ContactFilter2D {
            useLayerMask = true,
            layerMask = LayerMask.GetMask("Default"),
        };
    }

    public Vector2 GetMouseCommand() {
        Camera mainCamera = Camera.main;
        if (mainCamera == null || maxRange <= Mathf.Epsilon) {
            return Vector2.zero;
        }

        float depth = Mathf.Abs(mainCamera.transform.position.z);
        Vector3 center =
            new Vector3(Screen.width * 0.5f, Screen.height * 0.5f, depth);
        Vector3 mouse =
            new Vector3(Input.mousePosition.x, Input.mousePosition.y, depth);

        center = mainCamera.ScreenToWorldPoint(center);
        mouse = mainCamera.ScreenToWorldPoint(mouse);

        return Vector2.ClampMagnitude((mouse - center) / maxRange, 1.0f);
    }

    public void SetExternalControlEnabled(bool enabled) {
        externalControlEnabled = enabled;
        if (!enabled) {
            externalCommand = Vector2.zero;
        }
    }

    public void SetExternalCommand(Vector2 normalizedCommand) {
        externalCommand = Vector2.ClampMagnitude(normalizedCommand, 1.0f);
    }

    public void ResetControlState() {
        externalCommand = Vector2.zero;
        LastAppliedCommand = Vector2.zero;
        HammerTouchingTerrain = false;
    }

    public void ApplyNormalizedCommand(Vector2 normalizedCommand) {
        if (hammerRigidbody == null) {
            CacheComponents();
            InitializeContactFilter();
        }

        Vector2 command = Vector2.ClampMagnitude(normalizedCommand, 1.0f);
        Vector2 hammerOffset = command * maxRange;
        LastAppliedCommand = command;

        int contactCount = hammerCollider.OverlapCollider(
            terrainContactFilter, contactResults);
        HammerTouchingTerrain = false;
        for (int i = 0; i < contactCount; i++) {
            Collider2D candidate = contactResults[i];
            if (candidate != null &&
                !candidate.transform.IsChildOf(transform)) {
                HammerTouchingTerrain = true;
                break;
            }
        }

        if (HammerTouchingTerrain) {
            Vector2 targetBodyPosition =
                (Vector2)hammerHead.position - hammerOffset;
            Vector2 force =
                (targetBodyPosition - (Vector2)body.position) * 80.0f;
            bodyRigidbody.AddForce(force);
            bodyRigidbody.velocity =
                Vector2.ClampMagnitude(bodyRigidbody.velocity, 6.0f);
        }

        Vector2 desiredHammerPosition = (Vector2)body.position + hammerOffset;
        Vector2 smoothedHammerPosition = Vector2.Lerp(
            hammerHead.position, desiredHammerPosition, 0.2f);
        hammerRigidbody.MovePosition(smoothedHammerPosition);

        hammerHead.rotation = Quaternion.FromToRotation(
            Vector3.right, smoothedHammerPosition - (Vector2)body.position);
    }
}
