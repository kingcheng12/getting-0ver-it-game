using UnityEngine;

public class RLWaypoint : MonoBehaviour {
    [SerializeField, Min(0.05f)] float radius = 0.75f;

    public float Radius {
        get { return Mathf.Max(radius, 0.05f); }
    }

    public bool Contains(Vector2 position) {
        return Vector2.Distance(position, transform.position) <= Radius;
    }

    void OnDrawGizmos() {
        Gizmos.color = new Color(0.1f, 0.9f, 1.0f, 0.9f);
        Gizmos.DrawWireSphere(transform.position, Radius);
        Gizmos.DrawLine(
            transform.position + Vector3.left * Radius,
            transform.position + Vector3.right * Radius);
        Gizmos.DrawLine(
            transform.position + Vector3.down * Radius,
            transform.position + Vector3.up * Radius);
    }
}
