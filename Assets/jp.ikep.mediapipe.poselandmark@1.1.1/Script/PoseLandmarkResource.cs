using UnityEngine;
using Unity.Sentis;

namespace Mediapipe.PoseLandmark{
    [CreateAssetMenu(fileName = "PoseLandmark", menuName = "ScriptableObjects/Pose Landmark Resource")]
    public class PoseLandmarkResource : ScriptableObject
    {
        public ComputeShader preProcessCS;
        public ComputeShader postProcessCS;
        public ModelAsset liteModel;
        public ModelAsset fullModel;
    }
}