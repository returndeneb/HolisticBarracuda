using UnityEngine;
using UnityEditor;
using System.Reflection;

// Simple string label with GUIContent
struct Label
{
    GUIContent _guiContent;

    public static implicit operator GUIContent(Label label) => label._guiContent;

    public static implicit operator Label(string text)
        => new() { _guiContent = new GUIContent(text) };
}

// Auto-scanning serialized property wrapper
struct AutoProperty
{
    SerializedProperty _prop;

    public SerializedProperty Target => _prop;

    public AutoProperty(SerializedProperty prop)
        => _prop = prop;

    public static implicit operator
        SerializedProperty(AutoProperty prop) => prop._prop;

    public static void Scan<T>(T target) where T : Editor
    {
        var so = target.serializedObject;

        var flags = BindingFlags.Public | BindingFlags.NonPublic;
        flags |= BindingFlags.Instance;
            
        foreach (var f in typeof(T).GetFields(flags))
            if (f.FieldType == typeof(AutoProperty))
            {
                f.SetValue(target, new AutoProperty(so.FindProperty(f.Name)));
            }
                    
    }
}

public class WebCamInput : MonoBehaviour
{
    [SerializeField] string webCamName= "";
    [SerializeField] Vector2 webCamResolution = new Vector2(1920, 1080);
    [SerializeField] int webcamFrameRate = 30;
    [SerializeField] Texture staticInput;

    // Provide input image Texture.
    public Texture inputImageTexture{
        get{
            if(staticInput != null) return staticInput;
            return inputRT;
        }
    }

    WebCamTexture webCamTexture;
    RenderTexture inputRT;

    void Start()
    {
        if(staticInput == null){
            webCamTexture = new WebCamTexture(webCamName, (int)webCamResolution.x, (int)webCamResolution.y,webcamFrameRate);
            webCamTexture.Play();
        }

        inputRT = new RenderTexture((int)webCamResolution.x, (int)webCamResolution.y, 0);
    }

    void Update()
    {
        if(staticInput != null) return;
        if(!webCamTexture.didUpdateThisFrame) return;

        var aspect1 = (float)webCamTexture.width / webCamTexture.height;
        var aspect2 = (float)inputRT.width / inputRT.height;
        var aspectGap = aspect2 / aspect1;

        var vMirrored = webCamTexture.videoVerticallyMirrored;
        var scale = new Vector2(aspectGap, vMirrored ? -1 : 1);
        var offset = new Vector2((1 - aspectGap) / 2, vMirrored ? 1 : 0);

        Graphics.Blit(webCamTexture, inputRT, scale, offset);
    }

    void OnDestroy(){
        if (webCamTexture != null) Destroy(webCamTexture);
        if (inputRT != null) Destroy(inputRT);
    }
}


[CustomEditor(typeof(WebCamInput))]
sealed class WebCamList : Editor
{
    static class Labels
    {
        public static Label DeviceName = "Device Name";
        public static Label FrameRate = "Frame Rate";
        public static Label Resolution = "Resolution";
        public static Label Select = "Select";
    }


    AutoProperty webCamName;
    AutoProperty webCamResolution;
    AutoProperty webcamFrameRate;

    AutoProperty staticInput;

    void OnEnable() => AutoProperty.Scan(this);

    void ChangeWebcam(string name)
    {
        serializedObject.Update();
        webCamName.Target.stringValue = name;
        serializedObject.ApplyModifiedProperties();
    }

    void ShowDeviceSelector(Rect rect)
    {
        var menu = new GenericMenu();

        foreach (var device in WebCamTexture.devices)
            menu.AddItem(new GUIContent(device.name), false,
                () => ChangeWebcam(device.name));

        menu.DropDown(rect);
    }

    public override void OnInspectorGUI()
    {
        serializedObject.Update();

        EditorGUI.BeginDisabledGroup(Application.isPlaying);

        EditorGUI.indentLevel++;


        EditorGUILayout.BeginHorizontal();
        // Debug.Log(webcamName);
        EditorGUILayout.PropertyField(webCamName, Labels.DeviceName);
        var rect = EditorGUILayout.GetControlRect(false, GUILayout.Width(60));
        if (EditorGUI.DropdownButton(rect, Labels.Select, FocusType.Keyboard))
            ShowDeviceSelector(rect);
        EditorGUILayout.EndHorizontal();

        EditorGUILayout.PropertyField(webCamResolution, Labels.Resolution);
        EditorGUILayout.PropertyField(webcamFrameRate, Labels.FrameRate);
        

        EditorGUI.indentLevel--;

        EditorGUILayout.PropertyField(staticInput);

        EditorGUI.EndDisabledGroup();

        serializedObject.ApplyModifiedProperties();
    }
}