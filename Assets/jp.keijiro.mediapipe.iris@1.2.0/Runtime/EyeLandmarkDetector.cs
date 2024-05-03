using Unity.Sentis;
using UnityEngine;
using Klak.NNUtils;
using Klak.NNUtils.Extensions;

namespace MediaPipe.Iris {

public sealed class EyeLandmarkDetector : System.IDisposable
{
    ResourceSet _resources;
    IWorker _worker;
    ImagePreprocess _preprocess;
    GraphicsBuffer _output;
    BufferReader<Vector4> _readCache;
    
    public const int IrisVertexCount = 5;
    public const int ContourVertexCount = 71;
    public const int VertexCount = IrisVertexCount + ContourVertexCount;
    const int ImageSize = 64;
    
    public GraphicsBuffer VertexBuffer
        => _output;
    
    public System.ReadOnlySpan<Vector4> VertexArray
        => _readCache.Cached;

    public EyeLandmarkDetector(ResourceSet resources)
    {
        _resources = resources;
        
        _worker = WorkerFactory.CreateWorker(BackendType.GPUCompute, ModelLoader.Load(_resources.model));

        _preprocess = new ImagePreprocess(ImageSize, ImageSize, nchwFix: true);

        _output = BufferUtil.NewStructured<Vector4>(VertexCount);

        _readCache = new BufferReader<Vector4>(_output, VertexCount);
    }

    public void Dispose()
    {
        _worker?.Dispose();
        _worker = null;

        _preprocess?.Dispose();
        _preprocess = null;

        _output?.Dispose();
        _output = null;
    }

    public void ProcessImage(Texture image)
    {
        _preprocess.Dispatch(image, _resources.preprocess);
        _worker.Execute(_preprocess.Tensor);
        var post = _resources.postprocess;
        post.SetBuffer(0, "_IrisTensor", _worker.PeekOutputBuffer("output_iris"));
        post.SetBuffer(0, "_ContourTensor", _worker.PeekOutputBuffer("output_eyes_contours_and_brows"));
        post.SetBuffer(0, "_Vertices", _output);
        post.Dispatch(0, 1, 1, 1);
        _readCache.InvalidateCache();
    }
    
}

} 