package org.bitPurpleCloud;

import org.bitPurpleCloud.util.GetSharpness;
import org.opencv.core.*;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;


import java.util.ArrayList;
import java.util.List;


public class ImageSharpness {

    static { System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }

    public static void main(String[] args) {
        String imagePath = "src/main/resources/testJPG/bad.JPG";
        String modelWeights = "src/main/resources/model/yolov3.weights";
        String modelConfig = "src/main/resources/model/yolov3.cfg";

        Mat image = Imgcodecs.imread(imagePath);
        if (image.empty()) {
            System.out.println("Image not found or could not be loaded.");
            return;
        }

        // 加载YOLO模型
        Net net = Dnn.readNetFromDarknet(modelConfig, modelWeights);

        // 创建 blob 并设置为网络的输入
        Mat blob = Dnn.blobFromImage(image, 1 / 255.0, new Size(416, 416), new Scalar(0), true, false);
        net.setInput(blob);

        // 前向传播获取检测结果
        List<Mat> result = new ArrayList<>();
        List<String> outBlobNames = net.getUnconnectedOutLayersNames();
        net.forward(result, outBlobNames);

        // 存储检测框、置信度和类别ID
        List<Rect2d> boxes = new ArrayList<>();
        List<Float> confidences = new ArrayList<>();

        // 解析检测结果
        for (Mat level : result) {
            for (int i = 0; i < level.rows(); ++i) {
                Mat row = level.row(i);
                Mat scores = row.colRange(5, level.cols());
                Core.MinMaxLocResult mm = Core.minMaxLoc(scores);
                float confidence = (float) mm.maxVal;

                // 检测置信度阈值
                if (confidence > 0.5) {
                    float centerX = (float) (row.get(0, 0)[0] * image.width());
                    float centerY = (float) (row.get(0, 1)[0] * image.height());
                    float width = (float) (row.get(0, 2)[0] * image.width());
                    float height = (float) (row.get(0, 3)[0] * image.height());

                    double left = centerX - width / 2;
                    double top = centerY - height / 2;

                    Rect2d rect = new Rect2d(left, top, width, height);

                    // 添加到列表中
                    boxes.add(rect);
                    confidences.add(confidence);
                }
            }
        }

        // 应用非极大值抑制
        float nmsThreshold = 0.4f;
        MatOfFloat confidencesMat = new MatOfFloat(Converters.vector_float_to_Mat(confidences));
        MatOfRect2d boxesMat = new MatOfRect2d();
        boxesMat.fromList(boxes);
        MatOfInt indices = new MatOfInt();
        Dnn.NMSBoxes(boxesMat, confidencesMat, 0.5f, nmsThreshold, indices);

        // 绘制检测框并计算清晰度
        int[] indicesArray = indices.toArray();
        for (int idx : indicesArray) {
            Rect2d box = boxes.get(idx);

            // 确保 Rect 在图像范围内
            if (box.x >= 0 && box.y >= 0 &&
                    box.x + box.width <= image.width() &&
                    box.y + box.height <= image.height()) {

                // 计算清晰度评分
                Rect intBox = new Rect((int) box.x, (int) box.y, (int) box.width, (int) box.height);
                Mat subject = new Mat(image, intBox);
                double blurScore = GetSharpness.getQualityScore(subject);
                String blurText = String.format("Sharpness: %.2f", blurScore);

                // 在图像上绘制红框
                Imgproc.rectangle(image, intBox, new Scalar(0, 0, 255), 2);

                // 在框上方标记清晰度评分
                int textX = intBox.x;
                int textY = intBox.y - 10;  // 将文本放在框的上方
                Imgproc.putText(image, blurText, new Point(textX, textY), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(0, 0, 255), 1);
            }
        }

        // 保存和展示结果
        Imgcodecs.imwrite("src/main/resources/output/result_with_sharpness.jpg", image);
        System.out.println("Detection completed. Result saved to 'src/main/resources/output/result_with_sharpness.jpg'");
    }


}
