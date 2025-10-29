
using UnityEngine;
using UnityEngine.UI;

public class FitnessHUD : MonoBehaviour
{
    public Text label;
    public void Set(float bestF, float meanF)
    {
        if (label) label.text = $"Best: {bestF:F3}  | Mean: {meanF:F3}";
    }
}
