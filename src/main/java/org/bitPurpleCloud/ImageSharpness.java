package org.bitPurpleCloud;

import org.opencv.core.*;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

public class ImageSharpness {

    static { System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }

    public static void main(String[] args) {
        String imagePath = "src/main/resources/testJPG/good.JPG";
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

        // 解析检测结果并在图像上绘制红框和标记清晰度
        for (Mat detection : result) {
            for (int i = 0; i < detection.rows(); i++) {
                Mat row = detection.row(i);
                if (row.cols() > 5) {
                    Mat scores = row.colRange(5, row.cols());
                    Core.MinMaxLocResult mm = Core.minMaxLoc(scores);
                    float confidence = (float) mm.maxVal;

                    // 检测置信度阈值
                    if (confidence > 0.5) {
                        int centerX = (int) (row.get(0, 0)[0] * image.width());
                        int centerY = (int) (row.get(0, 1)[0] * image.height());
                        int width = (int) (row.get(0, 2)[0] * image.width());
                        int height = (int) (row.get(0, 3)[0] * image.height());

                        // 计算主体矩形框
                        Rect subjectRect = new Rect(centerX - width / 2, centerY - height / 2, width, height);

                        // 确保 Rect 在图像范围内
                        if (subjectRect.x >= 0 && subjectRect.y >= 0 &&
                                subjectRect.x + subjectRect.width <= image.width() &&
                                subjectRect.y + subjectRect.height <= image.height()) {

                            // 计算清晰度评分
                            Mat subject = new Mat(image, subjectRect);
                            double blurScore = calculateLaplacianVariance(subject);
                            String blurText = String.format("Sharpness: %.2f", blurScore);

                            // 在图像上绘制红框
                            Imgproc.rectangle(image, subjectRect, new Scalar(0, 0, 255), 2);

                            // 在框上方标记清晰度评分
                            int textX = subjectRect.x;
                            int textY = subjectRect.y - 10;  // 将文本放在框的上方
                            Imgproc.putText(image, blurText, new Point(textX, textY), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(0, 0, 255), 1);
                        }
                    }
                }
            }
        }

        // 保存和展示结果
        Imgcodecs.imwrite("src/main/resources/output/result_with_sharpness.jpg", image);
        System.out.println("Detection completed. Result saved to 'src/main/resources/output/result_with_sharpness.jpg'");
    }

    // 计算Laplacian方差，用于清晰度评分
    private static double calculateLaplacianVariance(Mat image) {
        Mat laplacian = new Mat();
        Imgproc.Laplacian(image, laplacian, CvType.CV_64F);

        MatOfDouble variance = new MatOfDouble();
        Core.meanStdDev(laplacian, new MatOfDouble(), variance);

        return Math.pow(variance.get(0, 0)[0], 2);
    }
}
