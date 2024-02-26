import rclpy

from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from .modules.image_quantizer import ImageQuantization

class ImageQuantizer(Node):
    def __init__(self, quantization_method, quantization_level):
        super().__init__('image_quantizer')
        self.sub = self.create_subscription(Image, 'cam0/image_raw', self.image_callback, 100)
        self.pub = self.create_publisher(Image, 'cam0/image_quantized', 100)
        
        self.method = quantization_method
        self.level  = quantization_level
        self.image_quantization = ImageQuantization()
        self.cv_bridge = CvBridge()


    def image_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        header = msg.header

        # Perform quantization
        if self.method == 'linear':
            quantized_image = self.image_quantization.linear_quantization(cv_image, N=self.level)
        if self.method == 'blockwise':
            quantized_image = self.image_quantization.blockwise_quantization(cv_image, N=self.level)
        if self.method == 'error_diffusion':
            quantized_image = self.image_quantization.error_diffusion_quantization(cv_image, N=self.level)

        # Convert image back to ROS Image message
        quantized_msg = self.cv_bridge.cv2_to_imgmsg(quantized_image)
        quantized_msg.header = header

        # Publish the quantized image
        self.pub.publish(quantized_msg)


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