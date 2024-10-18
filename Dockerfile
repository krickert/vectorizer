# Use an official Eclipse Temurin base image as Jib typically does for Java apps
FROM eclipse-temurin:17

# Set working directory
WORKDIR /app

# Copy dependencies (Jib often splits dependencies, resources, and class files)
COPY target/dependency/* /app/libs/

# Copy resource files
COPY src/main/resources /app/resources

# Copy classes and other files
COPY target/classes /app/classes

# Copy the executable JAR (if your application uses a JAR file)
COPY target/vectorizer-*.jar /app/vectorizer.jar

# Environment variables
ENV JAVA_HOME=/opt/java/openjdk
ENV PATH="/opt/java/openjdk/bin:${PATH}"

# Expose the application port
EXPOSE 60401

# The entry point for your application (you might see this in `docker inspect`)
ENTRYPOINT ["java", "-jar", "/app/vectorizer.jar"]

