using UnityEditor;
using UnityEngine;


[CustomEditor(typeof(WebCamInput))]
sealed class ImageSourceEditor : Editor
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
