
using UnityEngine;

public class BestReplay : MonoBehaviour
{
    public CreatureController ctrl;
    void Start()
    {
        TextAsset ta = Resources.Load<TextAsset>("best_dna");
        if (ta == null){ Debug.LogWarning("best_dna.json no encontrado en Resources"); return; }
        var dna = JsonUtility.FromJson<CreatureDNA>(ta.text);
        ctrl.dna = dna;
        ctrl.enabled = true;
    }
}
