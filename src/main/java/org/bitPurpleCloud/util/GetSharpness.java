package org.bitPurpleCloud.util;

import com.alibaba.fastjson2.JSON;
import com.alibaba.fastjson2.JSONException;
import com.alibaba.fastjson2.JSONObject;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;


import java.awt.image.BufferedImage;
import java.io.*;
import java.net.Socket;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;


public class GetSharpness {
    public static double calculateTenengradSharpness(Mat image) {
        Mat gray = new Mat();
        if (image.channels() == 3) {
            Imgproc.cvtColor(image, gray, Imgproc.COLOR_BGR2GRAY);
        } else {
            gray = image;
        }

        Mat gradX = new Mat();
        Mat gradY = new Mat();

        // 计算 Sobel 梯度
        Imgproc.Sobel(gray, gradX, CvType.CV_64F, 1, 0);
        Imgproc.Sobel(gray, gradY, CvType.CV_64F, 0, 1);

        Mat gradSquareX = new Mat();
        Mat gradSquareY = new Mat();

        // 计算梯度的平方
        Core.multiply(gradX, gradX, gradSquareX);
        Core.multiply(gradY, gradY, gradSquareY);

        Mat gradientMagnitude = new Mat();
        Core.add(gradSquareX, gradSquareY, gradientMagnitude);

        // 计算梯度幅值的均值
        Scalar meanGradient = Core.mean(gradientMagnitude);
        return meanGradient.val[0];
    }

    public static double calculateFFTSharpness(Mat image) {
        Mat gray = new Mat();
        if (image.channels() == 3) {
            Imgproc.cvtColor(image, gray, Imgproc.COLOR_BGR2GRAY);
        } else {
            gray = image;
        }

        // 将 gray 转换为 64F 深度，以便与其他矩阵深度一致
        Mat padded = new Mat();
        gray.convertTo(padded, CvType.CV_64F);

        // 确保尺寸适合 FFT
        int m = Core.getOptimalDFTSize(padded.rows());
        int n = Core.getOptimalDFTSize(padded.cols());
        Core.copyMakeBorder(padded, padded, 0, m - padded.rows(), 0, n - padded.cols(), Core.BORDER_CONSTANT, Scalar.all(0));

        // 创建复数矩阵
        Mat complexImage = new Mat();
        List<Mat> planes = new ArrayList<>();
        planes.add(padded);
        planes.add(Mat.zeros(padded.size(), CvType.CV_64F)); // 确保深度一致
        Core.merge(planes, complexImage);

        // 进行 DFT
        Core.dft(complexImage, complexImage);

        // 计算高频能量
        Mat mag = new Mat();
        List<Mat> newPlanes = new ArrayList<>();
        Core.split(complexImage, newPlanes);
        Core.magnitude(newPlanes.get(0), newPlanes.get(1), mag);

        Mat magMat = mag.submat(new Rect(0, 0, mag.cols() / 2, mag.rows() / 2));
        Scalar meanVal = Core.mean(magMat);
        return meanVal.val[0];
    }

    public static double calculateLaplacianEnergy(Mat image) {
        Mat gray = new Mat();
        if (image.channels() == 3) {
            Imgproc.cvtColor(image, gray, Imgproc.COLOR_BGR2GRAY);
        } else {
            gray = image;
        }

        Mat laplacian = new Mat();
        Imgproc.Laplacian(gray, laplacian, CvType.CV_64F);

        Mat absLaplacian = new Mat();
        Core.absdiff(laplacian, Scalar.all(0), absLaplacian);
        Scalar sumAbsLaplacian = Core.sumElems(absLaplacian);

        return sumAbsLaplacian.val[0];
    }
    public static double calculateCombinedSharpness(Mat image) {
        // Laplacian 方差
        double laplacianVariance = calculateLaplacianVariance(image);

        // Tenengrad 梯度幅值均值
        double tenengradSharpness = calculateTenengradSharpness(image);

        // 综合得分（可以按权重调整）
        return 0.7 * laplacianVariance + 0.3 * tenengradSharpness;
    }
    // 计算Laplacian方差，用于清晰度评分
    public static double calculateLaplacianVariance(Mat image) {
        Mat laplacian = new Mat();
        Imgproc.Laplacian(image, laplacian, CvType.CV_64F);

        MatOfDouble variance = new MatOfDouble();
        Core.meanStdDev(laplacian, new MatOfDouble(), variance);

        return Math.pow(variance.get(0, 0)[0], 2);
    }


    public static float getQualityScore(Mat image) {
        // 将Mat对象保存为临时文件
        String tempImagePath = "D:\\code\\java\\MyPixcake\\src\\main\\resources\\testJPG\\temp_image.jpg";
        Imgcodecs.imwrite(tempImagePath, image);

        // Python服务的地址和端口
        String serverHost = "localhost";
        int serverPort = 5000;

        try (Socket socket = new Socket(serverHost, serverPort);
             PrintWriter out = new PrintWriter(socket.getOutputStream(), true);
             BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()))) {

            // 发送图像路径到Python服务
            out.println(tempImagePath);

            // 等待接收来自Python服务的响应
            String response = in.readLine();
            System.out.println("Received from Python service: " + response);

            // 解析返回的JSON，提取mean_score_prediction
            return parseMeanScore(response);

        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            // 删除临时文件
            new File(tempImagePath).delete();
        }
        return -1; // 如果出错，返回-1
    }

    private static float parseMeanScore(String response) {
        try {
            // 使用 fastjson 来解析 JSON 字符串
            JSONObject json = JSON.parseObject(response);
            return json.getFloat("mean_score_prediction");
        } catch (Exception e) {
            e.printStackTrace();
        }
        return -1; // 如果解析出错，返回 -1
    }


}
