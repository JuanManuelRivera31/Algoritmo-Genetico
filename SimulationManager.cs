
using UnityEngine;
public class SimulationManager : MonoBehaviour
{
    public bool slowMo = false;
    void Update(){ Time.timeScale = slowMo ? 0.5f : 1f; }
}
