/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <sys/time.h>
#include <cuda_runtime_api.h>
#include <nvdsinfer.h>
#include "gstnvdsmeta.h"
// #include "gstnvstreammeta.h"
#ifndef PLATFORM_TEGRA
#include "gst-nvmessage.h"
#endif

#define MAX_DISPLAY_LEN 64

#define PGIE_CLASS_ID_VEHICLE 0
#define PGIE_CLASS_ID_PERSON 2

/* The muxer output resolution must be set if the input streams will be of
 * different resolution. The muxer will scale all the input frames to this
 * resolution. */
#define MUXER_OUTPUT_WIDTH 730
#define MUXER_OUTPUT_HEIGHT 247

/* Muxer batch formation timeout, for e.g. 40 millisec. Should ideally be set
 * based on the fastest source's framerate. */
#define MUXER_BATCH_TIMEOUT_USEC 40000

#define TILED_OUTPUT_WIDTH 730
#define TILED_OUTPUT_HEIGHT 247

/* NVIDIA Decoder source pad memory feature. This feature signifies that source
 * pads having this capability will push GstBuffers containing cuda buffers. */
#define GST_CAPS_FEATURES_NVMM "memory:NVMM"

#define NVINFER_PLUGIN "nvinfer"
#define NVINFERSERVER_PLUGIN "nvinferserver"
#define PGIE_CONFIG_FILE "dstest_image_decode_pgie_config.txt"
#define PGIE_NVINFERSERVER_CONFIG_FILE "dstest_image_decode_pgie_nvinferserver_config.txt"

gint pgie_classes_str[36] = {30, 35, 70, 75, 40, 45, 90, 95, 20, 25, 100, 130, 135, 170, 175, 140, 145, 120, 125, 110, 150, 105, 150, 155, 160, 165, 180, 10, 15, 5, 50, 55, 60, 65, 80, 85};

#define FPS_PRINT_INTERVAL 300

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

static gboolean
bus_call(GstBus *bus, GstMessage *msg, gpointer data)
{
  GMainLoop *loop = (GMainLoop *)data;
  switch (GST_MESSAGE_TYPE(msg))
  {
  case GST_MESSAGE_EOS:
    g_print("End of stream\n");
    g_main_loop_quit(loop);
    break;
  case GST_MESSAGE_WARNING:
  {
    gchar *debug;
    GError *error;
    gst_message_parse_warning(msg, &error, &debug);
    g_printerr("WARNING from element %s: %s\n",
               GST_OBJECT_NAME(msg->src), error->message);
    g_free(debug);
    g_printerr("Warning: %s\n", error->message);
    g_error_free(error);
    break;
  }
  case GST_MESSAGE_ERROR:
  {
    gchar *debug;
    GError *error;
    gst_message_parse_error(msg, &error, &debug);
    g_printerr("ERROR from element %s: %s\n",
               GST_OBJECT_NAME(msg->src), error->message);
    if (debug)
      g_printerr("Error details: %s\n", debug);
    g_free(debug);
    g_error_free(error);
    g_main_loop_quit(loop);
    break;
  }
#ifndef PLATFORM_TEGRA
  case GST_MESSAGE_ELEMENT:
  {
    if (gst_nvmessage_is_stream_eos(msg))
    {
      guint stream_id;
      if (gst_nvmessage_parse_stream_eos(msg, &stream_id))
      {
        g_print("Got EOS from stream %d\n", stream_id);
      }
    }
    break;
  }
#endif
  default:
    break;
  }
  return TRUE;
}

static GstElement *
create_source_bin(guint index, gchar *uri)
{
  GstElement *bin = NULL /*, *uri_decode_bin = NULL*/;
  gchar bin_name[16] = {};
  gboolean multi_file_src = FALSE;

  int current_device = -1;
  cudaGetDevice(&current_device);
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, current_device);

  g_snprintf(bin_name, 15, "source-bin-%02d", index);
  /* Create a source GstBin to abstract this bin's content from the rest of the
   * pipeline */
  bin = gst_bin_new(bin_name);

  GstElement *source, *jpegparser, *decoder;

  if (strstr(uri, "%d"))
  {
    source = gst_element_factory_make("multifilesrc", "source");
    multi_file_src = TRUE;
  }
  else
    source = gst_element_factory_make("filesrc", "source");

  jpegparser = gst_element_factory_make("jpegparse", "jpeg-parser");

  decoder = gst_element_factory_make("nvv4l2decoder", "nvv4l2-decoder");

  if (!source || !jpegparser || !decoder)
  {
    g_printerr("One element could not be created. Exiting.\n");
    return NULL;
  }

  g_object_set(G_OBJECT(source), "location", uri, NULL);

  const char *dot = strrchr(uri, '.');
  if ((!strcmp(dot + 1, "mjpeg")) || (!strcmp(dot + 1, "mjpg")) || (!strcmp(dot + 1, "mp4")) || (multi_file_src == TRUE))
  {
    if (prop.integrated)
    {
      g_object_set(G_OBJECT(decoder), "mjpeg", 1, NULL);
    }
  }

  gst_bin_add_many(GST_BIN(bin), source, jpegparser, decoder, NULL);

  gst_element_link_many(source, jpegparser, decoder, NULL);

  /* We need to create a ghost pad for the source bin which will act as a proxy
   * for the video decoder src pad. The ghost pad will not have a target right
   * now. Once the decode bin creates the video decoder and generates the
   * cb_newpad callback, we will set the ghost pad target to the video decoder
   * src pad. */
  if (!gst_element_add_pad(bin, gst_ghost_pad_new_no_target("src",
                                                            GST_PAD_SRC)))
  {
    g_printerr("Failed to add ghost pad in source bin\n");
    return NULL;
  }

  GstPad *srcpad = gst_element_get_static_pad(decoder, "src");
  if (!srcpad)
  {
    g_printerr("Failed to get src pad of source bin. Exiting.\n");
    return NULL;
  }
  GstPad *bin_ghost_pad = gst_element_get_static_pad(bin, "src");
  if (!gst_ghost_pad_set_target(GST_GHOST_PAD(bin_ghost_pad),
                                srcpad))
  {
    g_printerr("Failed to link decoder src pad to source bin ghost pad\n");
  }

  return bin;
}

static void
usage(const char *bin)
{
  g_printerr("Usage: %s  ...\n", bin);
  g_printerr("For nvinferserver, Usage: %s -t inferserver <elementary JPEG file1> <elementary JPEG file2> ...\n", bin);
}

int main(int argc, char *argv[])
{
  struct timeval start_time, end_time;
  double elapsed_time;

  // Bắt đầu đo thời gian
  gettimeofday(&start_time, NULL);
  GMainLoop *loop = NULL;
  GstElement *pipeline = NULL, *streammux = NULL, *sink = NULL, *nvvideoconv = NULL, *pgie = NULL,
             *nvvidconv = NULL, *nvosd = NULL, *tiler = NULL;
  GstBus *bus = NULL;
  guint bus_watch_id;
  GstPad *tiler_src_pad = NULL;
  guint i, num_sources = 100;
  guint tiler_rows, tiler_columns;
  guint pgie_batch_size;
  gboolean is_nvinfer_server = FALSE;
  const gchar *new_mux_str = g_getenv("USE_NEW_NVSTREAMMUX");
  gboolean use_new_mux = !g_strcmp0(new_mux_str, "yes");

  int current_device = -1;
  cudaGetDevice(&current_device);
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, current_device);

  /* Check input arguments */
  if (argc < 2)
  {
    usage(argv[0]);
    return -1;
  }

  if (argc >= 2 && !strcmp("-t", argv[1]))
  {
    if (!strcmp("inferserver", argv[2]))
    {
      is_nvinfer_server = TRUE;
    }
    else
    {
      usage(argv[0]);
      return -1;
    }
    g_print("Using nvinferserver as the inference plugin\n");
  }

  if (is_nvinfer_server)
  {
    num_sources = argc - 3;
  }
  else
  {
    num_sources = argc - 1;
  }

  /* Standard GStreamer initialization */
  gst_init(&argc, &argv);
  loop = g_main_loop_new(NULL, FALSE);

  /* Create gstreamer elements */
  /* Create Pipeline element that will form a connection of other elements */
  pipeline = gst_pipeline_new("dstest-image-decode-pipeline");

  /* Create nvstreammux instance to form batches from one or more sources. */
  streammux = gst_element_factory_make("nvstreammux", "stream-muxer");

  if (!pipeline || !streammux)
  {
    g_printerr("One element could not be created. Exiting.\n");
    return -1;
  }
  gst_bin_add(GST_BIN(pipeline), streammux);
  char image_path[100]; // Allocate enough space for the path
  for (i = 0; i < num_sources; i++)
  {
    GstPad *sinkpad, *srcpad;
    GstElement *source_bin;
    gchar pad_name[64] = {};
    /*if(i<10){
    sprintf(image_path, "/home/sonpro210802/tao-experiments/data/myntradataset/images/Image_000%d.jpg", i);
    argv[i + 1] = image_path;
    }else{
    sprintf(image_path, "/home/sonpro210802/tao-experiments/data/myntradataset/images/Image_00%d.jpg", i);
    argv[i + 1] = image_path;
    }*/
    if (is_nvinfer_server)
    {
      source_bin = create_source_bin(i, argv[i + 3]);
    }
    else
    {
      source_bin = create_source_bin(i, argv[i + 1]);
    }
    g_print("argv[%d]: %s\n", i + 1, argv[i + 1]);
    if (!source_bin)
    {
      g_printerr("Failed to create source bin. Exiting.\n");
      return -1;
    }

    gst_bin_add(GST_BIN(pipeline), source_bin);

    g_snprintf(pad_name, 16, "sink_%u", i);
    sinkpad = gst_element_request_pad_simple(streammux, pad_name);
    if (!sinkpad)
    {
      g_printerr("Streammux request sink pad failed. Exiting.\n");
      return -1;
    }

    srcpad = gst_element_get_static_pad(source_bin, "src");
    if (!srcpad)
    {
      g_printerr("Failed to get src pad of source bin. Exiting.\n");
      return -1;
    }

    if (gst_pad_link(srcpad, sinkpad) != GST_PAD_LINK_OK)
    {
      g_printerr("Failed to link source bin to stream muxer. Exiting.\n");
      return -1;
    }

    gst_object_unref(srcpad);
    gst_object_unref(sinkpad);
  }

  /* Use convertor to convert to appropriate format */
  nvvideoconv = gst_element_factory_make("nvvideoconvert", "nvvideo-converter1");

  /* Use nvinfer to infer on batched frame. */
  pgie = gst_element_factory_make(
      is_nvinfer_server ? NVINFERSERVER_PLUGIN : NVINFER_PLUGIN,
      "primary-nvinference-engine");

  /* Use nvtiler to composite the batched frames into a 2D tiled array based
   * on the source of the frames. */
  tiler = gst_element_factory_make("nvmultistreamtiler", "nvtiler");

  /* Use convertor to convert from NV12 to RGBA as required by nvosd */
  nvvidconv = gst_element_factory_make("nvvideoconvert", "nvvideo-converter");

  /* Create OSD to draw on the converted RGBA buffer */
  nvosd = gst_element_factory_make("nvdsosd", "nv-onscreendisplay");

  /* Finally render the osd output */
  if (prop.integrated)
  {
    sink = gst_element_factory_make("nv3dsink", "nvvideo-renderer");
  }
  else
  {
    sink = gst_element_factory_make("nveglglessink", "nvvideo-renderer");
  }
  g_object_set(G_OBJECT(sink), "sync", 0, NULL);

  if (!nvvideoconv || !pgie || !tiler || !nvvidconv || !nvosd || !sink)
  {
    g_printerr("One element could not be created. Exiting.\n");
    return -1;
  }

  if (!use_new_mux)
  {
    g_object_set(G_OBJECT(streammux), "width", MUXER_OUTPUT_WIDTH, "height",
                 MUXER_OUTPUT_HEIGHT, "batch-size", num_sources,
                 "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);
  }
  else
  {
    g_object_set(G_OBJECT(streammux), "batch-size", num_sources,
                 "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);
  }

  /* Configure the nvinfer/nvinferserver element using the config file. */
  if (is_nvinfer_server)
  {
    g_object_set(G_OBJECT(pgie), "config-file-path", PGIE_NVINFERSERVER_CONFIG_FILE, NULL);
  }
  else
  {
    g_object_set(G_OBJECT(pgie), "config-file-path", PGIE_CONFIG_FILE, NULL);
  }

  /* Override the batch-size set in the config file with the number of sources. */
  g_object_get(G_OBJECT(pgie), "batch-size", &pgie_batch_size, NULL);
  if (pgie_batch_size != num_sources)
  {
    g_printerr("WARNING: Overriding infer-config batch-size (%d) with number of sources (%d)\n",
               pgie_batch_size, num_sources);
    g_object_set(G_OBJECT(pgie), "batch-size", num_sources, NULL);
  }

  tiler_rows = (guint)sqrt(num_sources);
  tiler_columns = (guint)ceil(1.0 * num_sources / tiler_rows);
  /* we set the tiler properties here */
  g_object_set(G_OBJECT(tiler), "rows", tiler_rows, "columns", tiler_columns,
               "width", TILED_OUTPUT_WIDTH, "height", TILED_OUTPUT_HEIGHT, NULL);

  /* we add a message handler */
  bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
  bus_watch_id = gst_bus_add_watch(bus, bus_call, loop);
  gst_object_unref(bus);

  /* Set up the pipeline */
  /* we add all elements into the pipeline */
  gst_bin_add_many(GST_BIN(pipeline), nvvideoconv, pgie, tiler, nvvidconv, nvosd, sink,
                   NULL);
  /* we link the elements together
   * nvstreammux -> nvvideoconv -> pgie -> nvtiler -> nvvidconv -> nvosd -> video-renderer */
  if (!gst_element_link_many(streammux, nvvideoconv, pgie, tiler, nvvidconv, nvosd, sink,
                             NULL))
  {
    g_printerr("Elements could not be linked. Exiting.\n");
    return -1;
  }

  /* Lets add probe to get informed of the meta data generated, we add probe to
   * the sink pad of the osd element, since by that time, the buffer would have
   * had got all the metadata. */
  tiler_src_pad = gst_element_get_static_pad(pgie, "src");
  if (!tiler_src_pad)
    g_print("Unable to get src pad\n");
  else
    gst_pad_add_probe(tiler_src_pad, GST_PAD_PROBE_TYPE_BUFFER,
                      tiler_src_pad_buffer_probe, NULL, NULL);
  gst_object_unref(tiler_src_pad);

  /* Set the pipeline to "playing" state */
  g_print("Now playing...");
  gst_element_set_state(pipeline, GST_STATE_PLAYING);

  /* Wait till pipeline encounters an error or EOS */
  g_print("Running...\n");
  g_main_loop_run(loop);

  /* Out of the main loop, clean up nicely */
  g_print("Returned, stopping playback\n");
  gst_element_set_state(pipeline, GST_STATE_NULL);
  g_print("Deleting pipeline\n");
  gettimeofday(&end_time, NULL);

  // Tính toán thời gian đã trôi qua
  elapsed_time = (end_time.tv_sec - start_time.tv_sec) * 1000000.0;
  elapsed_time += (end_time.tv_usec - start_time.tv_usec);
  elapsed_time /= 1000000.0;

  printf("Elapsed time: %f seconds\n", elapsed_time);
  gst_object_unref(GST_OBJECT(pipeline));
  g_source_remove(bus_watch_id);
  g_main_loop_unref(loop);
  return 0;
}
