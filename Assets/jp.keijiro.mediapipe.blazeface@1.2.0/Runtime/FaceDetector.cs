using Unity.Sentis;
using UnityEngine;
using Klak.NNUtils;
using Klak.NNUtils.Extensions;

namespace MediaPipe.BlazeFace {

public sealed class FaceDetector : System.IDisposable
{
    ResourceSet _resources;
    int _size;
    IWorker _worker;
    ImagePreprocess _preprocess;
    (GraphicsBuffer post1, GraphicsBuffer post2, GraphicsBuffer count) _outputs;
    CountedBufferReader<Detection> _readCache;

    public FaceDetector(ResourceSet resources)
    {
        _resources = resources;

        var model = ModelLoader.Load(_resources.model);
        _size = model.inputs[0].GetTensorShape().GetWidth();

        _worker = WorkerFactory.CreateWorker(BackendType.GPUCompute, model);
        
        _preprocess = new ImagePreprocess(_size, _size);

        _outputs.post1 = new GraphicsBuffer(GraphicsBuffer.Target.Append, Detection.Max, Detection.Size);
        _outputs.post2 = new GraphicsBuffer(GraphicsBuffer.Target.Append, Detection.Max, Detection.Size);
        _outputs.count = new GraphicsBuffer(GraphicsBuffer.Target.Raw, 1, sizeof(uint));

        _readCache = new CountedBufferReader<Detection>(_outputs.post2, _outputs.count, Detection.Max);
    }

    public void Dispose()
    {
        _worker?.Dispose();
        _worker = null;

        _preprocess?.Dispose();
        _preprocess = null;

        _outputs.post1?.Dispose();
        _outputs.post2?.Dispose();
        _outputs.count?.Dispose();
        _outputs = (null, null, null);
    }

    public void ProcessImage(Texture image, float threshold = 0.75f)
    {
        _outputs.post1.SetCounterValue(0);
        _outputs.post2.SetCounterValue(0);

        _preprocess.Dispatch(image, _resources.preprocess);

        _worker.Execute(_preprocess.Tensor);

        var post1 = _resources.postprocess1;
        post1.SetFloat("_ImageSize", _size);
        post1.SetFloat("_Threshold", threshold);

        post1.SetBuffer(0, "_Scores", _worker.PeekOutputBuffer("Identity"));
        post1.SetBuffer(0, "_Boxes", _worker.PeekOutputBuffer("Identity_2"));
        post1.SetBuffer(0, "_Output", _outputs.post1);
        post1.Dispatch(0, 1, 1, 1);

        post1.SetBuffer(1, "_Scores", _worker.PeekOutputBuffer("Identity_1"));
        post1.SetBuffer(1, "_Boxes", _worker.PeekOutputBuffer("Identity_3"));
        post1.SetBuffer(1, "_Output", _outputs.post1);
        post1.Dispatch(1, 1, 1, 1);

        GraphicsBuffer.CopyCount(_outputs.post1, _outputs.count, 0);

        var post2 = _resources.postprocess2;
        post2.SetBuffer(0, "_Input", _outputs.post1);
        post2.SetBuffer(0, "_Count", _outputs.count);
        post2.SetBuffer(0, "_Output", _outputs.post2);
        post2.Dispatch(0, 1, 1, 1);

        GraphicsBuffer.CopyCount(_outputs.post2, _outputs.count, 0);

        _readCache.InvalidateCache();
    }

    public System.ReadOnlySpan<Detection> Detections
      => _readCache.Cached;
    
}

} // namespace MediaPipe.BlazeFace
