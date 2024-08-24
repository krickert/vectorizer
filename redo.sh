#!/bin/bash
#
mvn package -DskipTests=true
cd target/
java -agentlib:jdwp=transport=dt_socket,server=y,suspend=y,address=*:5005 -jar vectorizer-1.0-SNAPSHOT.jar 
cd -


