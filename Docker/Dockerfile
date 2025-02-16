# FROM python:3.12.7
# ADD requirements.txt /
# RUN pip install -r /requirements.txt
# RUN mkdir /docker_files/
# ADD ./docker_files/
# RUN chmod -R 777 /docker_files
# ENV PYTHONUNBUFFERED=1
# CMD [ "python", "./app.py" ]

FROM python:3.10
# Copy requirements.txt to the container
ADD requirements.txt /docker_files/
# Set working directory to /connect4_app
WORKDIR /docker_files
# Install dependencies
RUN pip install -r requirements.txt
# Copy the entire project folder into the container
ADD . /docker_files/
# Set permissions for the project folder
RUN chmod -R 777 /docker_files
# Set environment variable for unbuffered logs
ENV PYTHONUNBUFFERED=1
# Run the main Python file
CMD ["python", "app.py"]