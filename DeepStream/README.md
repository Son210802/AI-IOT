# Deploy with DeepStream

The following are instructions on how to deploy a model that determines the rotation angle and color of an object into DeepStream. DeepStream is a powerful and intelligent multi-stream video analysis AI engine, the GStreamer framework powers it. This guide only stops at putting the model into deepstream and printing the model results to the console screen, but in terms of processing speed, there are still many limitations.

## Guide

### Modify files config: 

Here we will put the model into the config file including 1 onnx file, 1 engine file, 1 label file along with related properties. [`file config`](https://github.com/Son210802/AI-IOT/blob/main/DeepStream/config/dstest_image_decode_pgie_config.txt)
![`file config`](https://github.com/Son210802/AI-IOT/blob/main/Image/fileconfig.jpg)

### Modify files C:
Edit the C file to get the prediction results from the pipeline to print to the console screen. In deepstrem, data about objects or frames at batch level is called metadata and it is stored through a buffer that can be passed between plugins thanks to sink and float elements. When it goes through the nvinfer plugin, the model will reason and nvinfer will return ClassifierMeta, ObjectMeta which are detected objects or objects that have been classified through the problem. This is a classfication model so we access Access the elements inside the ClassifierMeta structure to get the desired output and print it to the display screen.
Initially it will iterate each frame Frame_meta then iterates through the list of objects. Finally, we will traverse the list of detected classification classes in obj_meta->classifier_meta_list, get the data of the current classification class and save it to the l_class pointer. The final loop is to iterate through the list of detected labels in the current classification layer, then get the data of the current label using the pointer to access the classification data and save it to the label pointer in NvDsClassifierMeta. From there, to be able to get the results printed on the screen, we use the label pointer to access the class id of the classification class and the name of the classification class and we get the results when running the application.

```c
static GstPadProbeReturn
tiler_src_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer u_data)
{

  GstBuffer *buf = (GstBuffer *)info->data;
  static guint num_rects = 0;
  NvDsObjectMeta *obj_meta = NULL;
  guint vehicle_count = 0;
  guint person_count = 0;
  NvDsMetaList *l_frame = NULL;
  NvDsMetaList *l_obj = NULL;

  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);
  if (!batch_meta)
  {
    g_print("Error: Failed to get batch meta\n");
  }
  // Loop through frames in the batch
  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next)
  {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);
    if (!frame_meta)
    {
      g_print("Error: Failed to get frame meta\n");
      continue; // Skip processing this frame
    }
    // Loop through objects in the frame
    for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next)
    {
      obj_meta = (NvDsObjectMeta *)(l_obj->data);
      if (!obj_meta)
      {
        g_print("Error: Failed to get frame meta\n");
        continue; // Skip processing this object
      }
      // Assuming class ID is stored in obj_meta->class_id
      guint predicted_class_id = obj_meta->class_id;

      // Access label information using NvDsClassifierMeta (replace with your logic)
      const char *predicted_label = NULL;
      NvDsClassifierMeta *cmeta = NULL;
      NvDsLabelInfo *label = NULL;

      for (NvDsMetaList *l_class = obj_meta->classifier_meta_list; l_class != NULL; l_class = l_class->next)
      {
        cmeta = (NvDsClassifierMeta *)l_class->data;
        if (cmeta)
        {
          for (NvDsMetaList *l_label = cmeta->label_info_list; l_label != NULL; l_label = l_label->next)
          {
            label = (NvDsLabelInfo *)l_label->data;
            if (label)
            {
              // access the object's class to get the best predicted angle
              g_print("num_rects: %d, result_class_id: class= %d, result_label: %d degree\n",
                      num_rects, label->result_class_id, pgie_classes_str[label->result_class_id]);
              break;
            }
          }
          break;
        }
      }
    }
    num_rects++;
  }
  return GST_PAD_PROBE_OK;
}
```

## Result
![`result`](https://github.com/Son210802/AI-IOT/blob/main/Image/result-deepstream.png)
