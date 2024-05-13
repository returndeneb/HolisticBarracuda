using Unity.Barracuda;
using UnityEngine;

namespace MediaPipe.BlazePalm {

//
// Palm detector class
//
public sealed partial class PalmDetector : System.IDisposable
{
    // Maximum number of detections. This value must be matched with
    // MAX_DETECTION in Common.hlsl.
    const int MaxDetection = 64;

    Detection[] _post2ReadCache;

    ResourceSet _resources;
    ComputeBuffer _preBuffer;
    ComputeBuffer _post1Buffer;
    ComputeBuffer _post2Buffer;
    ComputeBuffer _countBuffer;
    IWorker _worker;
    int _size;
    
    public ComputeBuffer DetectionBuffer
      => _post2Buffer;

    public ComputeBuffer CountBuffer
      => _countBuffer;
    
    public PalmDetector(ResourceSet resources)
    {
        _resources = resources;
        var model = ModelLoader.Load(_resources.model);
        _size = model.inputs[0].shape[6]; // Input tensor width

        _preBuffer = new ComputeBuffer(_size * _size * 3, sizeof(float));

        _post1Buffer = new ComputeBuffer
            (MaxDetection, Detection.Size, ComputeBufferType.Append);

        _post2Buffer = new ComputeBuffer
            (MaxDetection, Detection.Size, ComputeBufferType.Append);

        _countBuffer = new ComputeBuffer
            (1, sizeof(uint), ComputeBufferType.Raw);

        _worker = model.CreateWorker();
    }

    public void Dispose()
    {
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
        // Preprocessing
        var pre = _resources.preprocess;
        pre.SetInt("_ImageSize", _size);
        pre.SetTexture(0, "_Texture", image);
        pre.SetBuffer(0, "_Tensor", _preBuffer);
        pre.Dispatch(0, _size / 8, _size / 8, 1);
        
        // Reset the compute buffer counters.
        _post1Buffer.SetCounterValue(0);
        _post2Buffer.SetCounterValue(0);

        // Run the BlazePalm model.
        using (var tensor = new Tensor(1, _size, _size, 3, _preBuffer))
            _worker.Execute(tensor);

        // Output tensors -> Temporary render textures
        var scoresRT = _worker.CopyOutputToTempRT("classificators",  1, 896);
        var  boxesRT = _worker.CopyOutputToTempRT("regressors"    , 18, 896);

        // 1st postprocess (bounding box aggregation)
        var post1 = _resources.postprocess1;
        post1.SetFloat("_ImageSize", _size);
        post1.SetFloat("_Threshold", threshold);

        post1.SetTexture(0, "_Scores", scoresRT);
        post1.SetTexture(0, "_Boxes", boxesRT);
        post1.SetBuffer(0, "_Output", _post1Buffer);
        post1.Dispatch(0, 1, 1, 1);

        post1.SetTexture(1, "_Scores", scoresRT);
        post1.SetTexture(1, "_Boxes", boxesRT);
        post1.SetBuffer(1, "_Output", _post1Buffer);
        post1.Dispatch(1, 1, 1, 1);

        // Release the temporary render textures.
        RenderTexture.ReleaseTemporary(scoresRT);
        RenderTexture.ReleaseTemporary(boxesRT);

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
