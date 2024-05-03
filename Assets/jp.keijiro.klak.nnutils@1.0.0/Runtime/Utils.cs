using Unity.Sentis;
using UnityEngine;

namespace Klak.NNUtils {

public static class RTUtil
{
    public static RenderTexture NewArgbUav(int w, int h)
    {
        var rt = new RenderTexture(w, h, 0, RenderTextureFormat.Default,
                                            RenderTextureReadWrite.Linear);
        rt.enableRandomWrite = true;
        rt.Create();
        return rt;
    }

    public static void Destroy(Object o)
    {
        if (o == null) return;
        if (Application.isPlaying)
            Object.Destroy(o);
        else
            Object.DestroyImmediate(o);
    }
}

public static class BufferUtil
{
    public unsafe static GraphicsBuffer NewStructured<T>(int length) where T : unmanaged
      => new GraphicsBuffer(GraphicsBuffer.Target.Structured, length, sizeof(T));

    public static (Tensor, ComputeTensorData) NewTensor(TensorShape shape, string name)
    {
        var data = new ComputeTensorData(shape, false);
        var tensor = TensorFloat.Zeros(shape);
        tensor.AttachToDevice(data);

        return (tensor, data);
    }
}

} // namespace Klak.NNUtils
