import boto3

class snsSender:
    def __init__(self):
        self.sns_client = boto3.client('sns', region_name='us-east-1')  # 지역 수정
        self.SNS_TOPIC_ARN = 'arn:aws:sns:us-east-1:YOUR_ACCOUNT_ID:FallDetectionTopic'  # ARN 수정
    
    def send_sns_message(self, message, subject):
        try:
            self.sns_client.publish(
                TopicArn=self.SNS_TOPIC_ARN,
                Message=message,
                Subject=subject
            )
        except Exception as e:
            print(f"Failed to send SNS notification: {e}")