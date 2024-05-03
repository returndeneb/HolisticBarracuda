using Unity.Sentis;
using UnityEngine;
using Klak.NNUtils;
using Klak.NNUtils.Extensions;

namespace MediaPipe.FaceLandmark {

public sealed class FaceLandmarkDetector : System.IDisposable
{
    ResourceSet _resources;
    IWorker _worker;
    ImagePreprocess _preprocess;
    GraphicsBuffer _output;
    BufferReader<Vector4> _readCache;

    public const int VertexCount = 468;
    const int ImageSize = 192;
    
    public GraphicsBuffer VertexBuffer
      => _output;

    public System.ReadOnlySpan<Vector4> VertexArray
      => _readCache.Cached;
  
    public FaceLandmarkDetector(ResourceSet resources)
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
      post.SetBuffer(0, "_Tensor", _worker.PeekOutputBuffer());
      post.SetBuffer(0, "_Vertices", _output);
      post.DispatchThreads(0, VertexCount, 1, 1);

      _readCache.InvalidateCache();
    }
}
} 
