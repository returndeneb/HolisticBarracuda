using UnityEngine;
using Unity.Sentis;

namespace Mediapipe.PoseDetection{
    public class PoseDetecter: System.IDisposable
    {
        #region public variable
        // Pose detection result buffer.
        public ComputeBuffer outputBuffer;
        // Pose detection result count buffer.
        public ComputeBuffer countBuffer;
        #endregion

        #region constant number 
        // Input image size defined by pose detection network model.
        const int IMAGE_SIZE = 128;
        // MAX_DETECTION must be matched with "Postprocess2.compute"
        const int MAX_DETECTION = 64;
        #endregion

        #region private variable
        IWorker woker;
        Model model;
        ComputeShader preProcessCS;
        ComputeShader postProcessCS;
        ComputeShader postProcess2CS;
        ComputeBuffer networkInputBuffer;
        ComputeBuffer postProcessBuffer;
        #endregion

        #region public method
        public PoseDetecter(PoseDetectionResource resource){
            preProcessCS = resource.preProcessCS;
            postProcessCS = resource.postProcessCS;
            postProcess2CS = resource.postProcess2CS;
            
            outputBuffer = new ComputeBuffer(MAX_DETECTION, PoseDetection.SIZE, ComputeBufferType.Append);
            countBuffer = new ComputeBuffer(1, sizeof(uint), ComputeBufferType.Raw);
            networkInputBuffer = new ComputeBuffer(IMAGE_SIZE * IMAGE_SIZE * 3, sizeof(float));
            postProcessBuffer = new ComputeBuffer(MAX_DETECTION, PoseDetection.SIZE, ComputeBufferType.Append);

            // Prepare neural network model.
            model = ModelLoader.Load(resource.model);
            woker = WorkerFactory.CreateWorker(BackendType.GPUCompute, model);
        }

        public void Dispose(){
            outputBuffer.Dispose();
            countBuffer.Dispose();
            networkInputBuffer.Dispose();
            postProcessBuffer.Dispose();
            woker.Dispose();
        }

        public void ProcessImage(Texture inputTexture, float poseThreshold = 0.75f, float iouThreshold = 0.3f){
            // Resize `inputTexture` texture to network model image size.
            preProcessCS.SetTexture(0, "_inputTexture", inputTexture);
            preProcessCS.SetBuffer(0, "_output", networkInputBuffer);
            preProcessCS.Dispatch(0, IMAGE_SIZE / 8, IMAGE_SIZE / 8, 1);

            ProcessImage(networkInputBuffer, poseThreshold, iouThreshold);
        }

        public void ProcessImage(ComputeBuffer input, float poseThreshold = 0.75f, float iouThreshold = 0.3f){
            // Reset append type buffer datas of previous frame. 
            postProcessBuffer.SetCounterValue(0);
            outputBuffer.SetCounterValue(0);

            //Execute neural network model.
            const int bufferSize = IMAGE_SIZE * IMAGE_SIZE * 3;
            var data = new float[bufferSize];
            input.GetData(data);
            var shape = new TensorShape(1, 3,IMAGE_SIZE, IMAGE_SIZE);
            var inputTensor = new TensorFloat(shape, data);
            woker.Execute(inputTensor);
            inputTensor.Dispose();

            //Get neural network model raw output as RenderTexture;
            // var scores = CopyOutputToTempRT("classificators", 1, 896);
            var scores = ((ComputeTensorData)woker.PeekOutput("classificators").tensorOnDevice).buffer;
            // var boxs = CopyOutputToTempRT("regressors", 12, 896);
            var boxs = ((ComputeTensorData)woker.PeekOutput("regressors").tensorOnDevice).buffer;
            // Parse raw result datas for above values of vectors.
            postProcessCS.SetFloat("_threshold", poseThreshold);
            postProcessCS.SetBuffer(0, "_scores", scores);
            postProcessCS.SetBuffer(0, "_boxs", boxs);
            postProcessCS.SetBuffer(0, "_output", postProcessBuffer);
            postProcessCS.Dispatch(0, 1, 1, 1);

            // Parse raw result datas for behind values of vectors.
            postProcessCS.SetBuffer(1, "_scores", scores);
            postProcessCS.SetBuffer(1, "_boxs", boxs);
            postProcessCS.SetBuffer(1, "_output", postProcessBuffer);
            postProcessCS.Dispatch(1, 1, 1, 1);

            // RenderTexture.ReleaseTemporary(scores);
            // RenderTexture.ReleaseTemporary(boxs);
            ComputeBuffer.CopyCount(postProcessBuffer, countBuffer, 0);
            
            // Get final results of pose deteciton.
            postProcess2CS.SetFloat("_iouThreshold", iouThreshold);
            postProcess2CS.SetBuffer(0, "_inputBuffer", postProcessBuffer);
            postProcess2CS.SetBuffer(0, "_inputCountBuffer", countBuffer);
            postProcess2CS.SetBuffer(0, "_output", outputBuffer);
            postProcess2CS.Dispatch(0, 1, 1, 1);

            // Set pose detection results count.
            ComputeBuffer.CopyCount(outputBuffer, countBuffer, 0);
        }
        #endregion

        #region private method
        // Exchange network output tensor to RenderTexture.
        RenderTexture CopyOutputToTempRT(string name, int w, int h)
        {
            var rtFormat = RenderTextureFormat.RFloat;
            var shape = new TensorShape(1, h, w, 1);
            var rt = RenderTexture.GetTemporary(w, h, 0, rtFormat);
            var tensor = woker.PeekOutput(name).ShallowReshape(shape) as TensorFloat;
            TextureConverter.RenderToTexture(tensor, rt);
            tensor.Dispose();
            return rt;
        }
        #endregion
    }
}
