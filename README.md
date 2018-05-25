To run the code on AWS

1. Connecting to the server:
open terminal:
cd /Users/chenjialu/Desktop/AWS/ (path of the key file)
chmod 0400 jialuc.pem
ssh -L localhost:8888:localhost:8888 -i jialuc.pem ubuntu@ec2-54-206-49-73.ap-southeast-2.compute.amazonaws.com
2. Run the code
source activate tensorflow_p36
cd Code
