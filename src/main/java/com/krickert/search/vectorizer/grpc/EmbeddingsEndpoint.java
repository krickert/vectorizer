package com.krickert.search.vectorizer.grpc;

import com.krickert.search.service.*;
import com.krickert.search.vectorizer.Vectorizer;
import io.grpc.stub.StreamObserver;
import io.micronaut.scheduling.TaskExecutors;
import jakarta.inject.Named;
import jakarta.inject.Singleton;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.*;
import java.util.stream.Collectors;

/**
 * This class represents the implementation of the gRPC endpoint for creating embeddings vectors.
 * It extends the abstract class EmbeddingServiceImplBase, which is the base class for the server implementation
 * of the EmbeddingService.
 */
@Singleton
public class EmbeddingsEndpoint extends EmbeddingServiceGrpc.EmbeddingServiceImplBase {
    private static final int BATCH_SIZE = 100; // Adjust the batch size based on your need

    private final Vectorizer vectorizer;
    private final ExecutorService batchExecutor;

    public EmbeddingsEndpoint(Vectorizer vectorizer, @Named(TaskExecutors.IO) ExecutorService batchExecutor) {
        this.vectorizer = vectorizer;
        this.batchExecutor = batchExecutor;
    }

    /**
     * Creates embeddings vector based on the given request and sends the reply to the response observer.
     */
    @Override
    public void createEmbeddingsVector(EmbeddingsVectorRequest request, StreamObserver<EmbeddingsVectorReply> responseObserver) {
        EmbeddingsVectorReply reply = createEmbeddingsReply(request.getText());
        sendReply(responseObserver, reply);
    }

    /**
     * Creates embeddings vectors based on the given request and sends the reply to the response observer.
     */
    @Override
    public void createEmbeddingsVectors(EmbeddingsVectorsRequest request, StreamObserver<EmbeddingsVectorsReply> responseObserver) {
        try {
            List<String> texts = request.getTextList();
            List<Future<List<EmbeddingsVectorReply>>> futures = new ArrayList<>();

            // Divide the list into batches and process
            for (int i = 0; i < texts.size(); i += BATCH_SIZE) {
                int end = Math.min(i + BATCH_SIZE, texts.size());
                List<String> batch = texts.subList(i, end);
                futures.add(batchExecutor.submit(() -> processBatch(batch)));
            }

            List<EmbeddingsVectorReply> allReplies = new ArrayList<>();
            for (Future<List<EmbeddingsVectorReply>> future : futures) {
                allReplies.addAll(future.get());
            }

            EmbeddingsVectorsReply.Builder builder = EmbeddingsVectorsReply.newBuilder();
            builder.addAllEmbeddings(allReplies);
            sendReply(responseObserver, builder.build());
        } catch (InterruptedException | ExecutionException e) {
            responseObserver.onError(e);
        }
    }

    private EmbeddingsVectorReply createEmbeddingsReply(String text) {
        Collection<Float> embeddings = vectorizer.getEmbeddings(text);
        EmbeddingsVectorReply.Builder builder = EmbeddingsVectorReply.newBuilder();
        builder.addAllEmbeddings(embeddings);
        return builder.build();
    }

    private List<EmbeddingsVectorReply> processBatch(List<String> texts) {
        return texts.stream()
                .map(this::createEmbeddingsReply)
                .collect(Collectors.toList());
    }

    @Override
    public void check(HealthCheckRequest request, StreamObserver<HealthCheckReply> responseObserver) {
        HealthCheckReply reply = HealthCheckReply.newBuilder().setStatus("EmbeddingService is running").build();
        sendReply(responseObserver, reply);
    }

    /**
     * Sends the reply to the response observer.
     */
    private <T> void sendReply(StreamObserver<T> responseObserver, T reply) {
        responseObserver.onNext(reply);
        responseObserver.onCompleted();
    }
}