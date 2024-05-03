using Unity.Sentis;
using UnityEngine;

namespace MediaPipe.BlazePalm {

static class IWorkerExtensions
{
    //
    // Retrieves an output tensor from a NN worker and returns it as a
    // temporary render texture. The caller must release it using
    // RenderTexture.ReleaseTemporary.
    //
    public static RenderTexture
      CopyOutputToTempRT(this IWorker worker, string name, int w, int h)
    {
        var fmt = RenderTextureFormat.RFloat;
        var shape = new TensorShape(1, h, w, 1);
        var rt = RenderTexture.GetTemporary(w, h, 0, fmt);
        using (var tensor = worker.PeekOutput(name).ShallowReshape(shape) as TensorFloat)
            TextureConverter.RenderToTexture(tensor, rt);
        return rt;
    }
}

} // namespace MediaPipe.BlazePalm
