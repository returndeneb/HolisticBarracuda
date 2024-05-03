using Klak.NNUtils;
using Klak.NNUtils.Extensions;
using Unity.Sentis;
using UnityEngine;

namespace MediaPipe.BlazePalm {

//
// Palm detector class
//
public sealed partial class PalmDetector : System.IDisposable
{
    ImagePreprocess _preprocess;
    ResourceSet _resources;
    ComputeBuffer _preBuffer;
    ComputeBuffer _post1Buffer;
    ComputeBuffer _post2Buffer;
    ComputeBuffer _countBuffer;
    IWorker _worker;
    int _size;
    Detection[] _post2ReadCache;
    const int MaxDetection = 64;
    
    public ComputeBuffer DetectionBuffer
      => _post2Buffer;

    public ComputeBuffer CountBuffer
      => _countBuffer;
    
    public PalmDetector(ResourceSet resources)
    {
        _resources = resources;
        var model = ModelLoader.Load(_resources.model);
        _size = model.inputs[0].GetTensorShape().GetWidth();
        _worker = WorkerFactory.CreateWorker(BackendType.GPUCompute, model);
        
        _preprocess = new ImagePreprocess(_size, _size, nchwFix: true);

        _preBuffer = new ComputeBuffer(_size * _size * 3, sizeof(float));

        _post1Buffer = new ComputeBuffer
            (MaxDetection, Detection.Size, ComputeBufferType.Append);

        _post2Buffer = new ComputeBuffer
            (MaxDetection, Detection.Size, ComputeBufferType.Append);

        _countBuffer = new ComputeBuffer
            (1, sizeof(uint), ComputeBufferType.Raw);
        
    }

    public void Dispose()
    { 
        _preprocess?.Dispose();
        _preprocess = null;
        
        _preBuffer?.Dispose();
        _preBuffer = null;

        _post1Buffer?.Dispose();
        _post1Buffer = null;

        _post2Buffer?.Dispose();
        _post2Buffer = null;

        _countBuffer?.Dispose();
        _countBuffer = null;

        _worker?.Dispose();
        _worker = null;
    }

    public void ProcessImage(Texture image, float threshold = 0.75f)
    {
        // var pre = _resources.preprocess;
        // pre.SetInt("_ImageSize", _size);
        // pre.SetTexture(0, "_Texture", image);
        // pre.SetBuffer(0, "_Tensor", _preBuffer);
        // pre.Dispatch(0, _size / 8, _size / 8, 1);
        
        _post1Buffer.SetCounterValue(0);
        _post2Buffer.SetCounterValue(0);
        
        // int bufferSize = _size * _size * 3;
        // var data = new float[bufferSize];
        // _preBuffer.GetData(data);
        // var shape = new TensorShape(1, 3,_size, _size);
        // var inputTensor = new TensorFloat(shape, data);
        // _worker.Execute(inputTensor);
        
        _worker.Execute(_preprocess.Tensor);
        
        var post1 = _resources.postprocess1;
        post1.SetFloat("_ImageSize", _size);
        post1.SetFloat("_Threshold", threshold);

        post1.SetBuffer(0, "_Scores", ((ComputeTensorData)_worker.PeekOutput("classificators").tensorOnDevice).buffer);
        post1.SetBuffer(0, "_Boxes", ((ComputeTensorData)_worker.PeekOutput("regressors").tensorOnDevice).buffer);
        post1.SetBuffer(0, "_Output", _post1Buffer);
        post1.Dispatch(0, 1, 1, 1);

        post1.SetBuffer(1, "_Scores", ((ComputeTensorData)_worker.PeekOutput("classificators").tensorOnDevice).buffer);
        post1.SetBuffer(1, "_Boxes", ((ComputeTensorData)_worker.PeekOutput("regressors").tensorOnDevice).buffer);
        post1.SetBuffer(1, "_Output", _post1Buffer);
        post1.Dispatch(1, 1, 1, 1);

        // Release the temporary render textures.

        // Retrieve the bounding box count.
        ComputeBuffer.CopyCount(_post1Buffer, _countBuffer, 0);

        // 2nd postprocess (overlap removal)
        var post2 = _resources.postprocess2;
        post2.SetBuffer(0, "_Input", _post1Buffer);
        post2.SetBuffer(0, "_Count", _countBuffer);
        post2.SetBuffer(0, "_Output", _post2Buffer);
        post2.Dispatch(0, 1, 1, 1);

        // Retrieve the bounding box count after removal.
        ComputeBuffer.CopyCount(_post2Buffer, _countBuffer, 0);

        // Read cache invalidation
        _post2ReadCache = null;
    }
    
}

} // namespace MediaPipe.BlazePalm
