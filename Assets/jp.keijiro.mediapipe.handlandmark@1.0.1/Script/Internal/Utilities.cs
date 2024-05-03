using Unity.Sentis;
using UnityEngine;

namespace MediaPipe.HandLandmark {

static class IWorkerExtensions
{
    // Peek a compute buffer of a worker output tensor.
    public static ComputeBuffer PeekOutputBuffer
      (this IWorker worker, string name)
      => ((ComputeTensorData)worker.PeekOutput(name).tensorOnDevice).buffer;
}

} // namespace MediaPipe.HandLandmark
