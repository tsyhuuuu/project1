import time, threading
import rclpy

from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


from .modules.image_quantizer import ImageQuantization
from .modules.timestamp2second import ts2sec

class ImageQuantizer(Node):
    def __init__(self, quantization_method, quantization_level):
        super().__init__('image_quantizer')

        self.sub_left  = self.create_subscription(Image, 'cam0/image_raw', self.image_callback_left, 100)
        self.sub_right = self.create_subscription(Image, 'cam1/image_raw', self.image_callback_right, 100)
        self.pub_left  = self.create_publisher(Image, 'cam0/image_quantized', 10)
        self.pub_right = self.create_publisher(Image, 'cam1/image_quantized', 10)
        
        self.method = quantization_method
        self.level  = quantization_level
        self.msgs   = [[], []]

        self.image_quantization = ImageQuantization()
        self.cv_bridge = CvBridge()

        sync_thread = threading.Thread(target=self.sync_both_images)
        sync_thread.start()


    def image_callback_left(self, msg):
        # Convert ROS Image message to OpenCV image
        cv_image_left = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        header_left = msg.header

        # Perform quantization
        if self.method == 'linear':
            quantized_image = self.image_quantization.linear_quantization(cv_image_left, N=self.level)
        if self.method == 'blockwise':
            quantized_image = self.image_quantization.blockwise_quantization(cv_image_left, N=self.level)
        if self.method == 'error_diffusion':
            quantized_image = self.image_quantization.error_diffusion_quantization(cv_image_left, N=self.level)

        # Convert image back to ROS Image message
        quantized_msg = self.cv_bridge.cv2_to_imgmsg(quantized_image)
        quantized_msg.header = header_left

        self.msgs[0].append(quantized_msg)
    

    def image_callback_right(self, msg):
        # Convert ROS Image message to OpenCV image
        cv_image_right = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        header_right = msg.header

        # Perform quantization
        if self.method == 'linear':
            quantized_image = self.image_quantization.linear_quantization(cv_image_right, N=self.level)
        if self.method == 'blockwise':
            quantized_image = self.image_quantization.blockwise_quantization(cv_image_right, N=self.level)
        if self.method == 'error_diffusion':
            quantized_image = self.image_quantization.error_diffusion_quantization(cv_image_right, N=self.level)

        # Convert image back to ROS Image message
        quantized_msg = self.cv_bridge.cv2_to_imgmsg(quantized_image)
        quantized_msg.header = header_right

        self.msgs[1].append(quantized_msg)
    

    def sync_both_images(self, max_time_diff=0.01):
        
        t_image_left, t_image_right = 0, 0
        while True:
            if not self.msgs[0].empty() and not self.msgs[1].empty():
                t_image_left = ts2sec(self.msgs[0][0].header.stamp)
                t_image_right = ts2sec(self.msgs[1][0].header.stamp)

                # Synchronize image buffers
                while (t_image_right - t_image_left) > max_time_diff and len(self.msgs[0]) > 1:
                    self.msgs[0].pop(0)
                    t_image_left = ts2sec(self.msgs[0].header.stamp)

                while (t_image_left - t_image_right) > max_time_diff and len(self.msgs[1]) > 1:
                    self.msgs[1].pop(0)
                    t_image_right = ts2sec(self.msgs[1].header.stamp)

                # Check for big time difference
                if (t_image_left - t_image_right) > max_time_diff or (t_image_right - t_image_left) > max_time_diff:
                    print("big time difference")
                    continue

                # Get images from buffers
                left_quantized_image_msg  = self.msgs[0].pop(0)
                right_quantized_image_msg = self.msgs[1].pop(0)

                # Sleep for a short duration
                time.sleep(0.001)

                # Publish images
                self.pub_left.publish(left_quantized_image_msg)
                self.pub_right.publish(right_quantized_image_msg)



def main(args=None, quantization_method='linear', quantization_level=2):
    rclpy.init(args=args)
    image_quantizer = ImageQuantizer(quantization_method, quantization_level)
    rclpy.spin(image_quantizer)
    image_quantizer.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        pass