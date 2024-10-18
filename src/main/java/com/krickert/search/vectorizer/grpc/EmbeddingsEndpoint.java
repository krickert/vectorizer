package com.krickert.search.vectorizer.grpc;

import com.krickert.search.service.*;
import com.krickert.search.vectorizer.Vectorizer;
import io.grpc.stub.StreamObserver;
import jakarta.inject.Singleton;
import reactor.core.publisher.Flux;
import reactor.core.scheduler.Schedulers;
import reactor.core.publisher.Mono;

import java.util.List;
import java.util.Optional;
import java.util.concurrent.Semaphore;

/**
 * This class represents the implementation of the gRPC endpoint for creating embeddings vectors.
 * It extends the abstract class EmbeddingServiceImplBase, which is the base class for the server implementation
 * of the EmbeddingService.
 */
@Singleton
public class EmbeddingsEndpoint extends EmbeddingServiceGrpc.EmbeddingServiceImplBase {
    private final Vectorizer vectorizer;
    private final Semaphore semaphore;

    public EmbeddingsEndpoint(Vectorizer vectorizer) {
        this.vectorizer = vectorizer;
        this.semaphore = new Semaphore(10); // Limit the number of concurrent requests to match predictor pool size
    }

    /**
     * Creates embeddings vector based on the given request and sends the reply to the response observer.
     */
    @Override
    public void createEmbeddingsVector(EmbeddingsVectorRequest request, StreamObserver<EmbeddingsVectorReply> responseObserver) {
        Mono.fromCallable(() -> {
                    acquireSemaphore();
                    return vectorizer.embeddings(request.getText(), Optional.empty());
                })
                .doOnTerminate(this::releaseSemaphore)
                .map(this::createEmbeddingsReply)
                .subscribeOn(Schedulers.boundedElastic())
                .subscribe(reply -> sendReply(responseObserver, reply), responseObserver::onError);
    }

    @Override
    public void createEmbeddingsVectors(EmbeddingsVectorsRequest request, StreamObserver<EmbeddingsVectorsReply> responseObserver) {
        List<String> texts = request.getTextList();
        EmbeddingsVectorsReply.Builder builder = EmbeddingsVectorsReply.newBuilder();

        Flux.fromIterable(texts)
                .flatMap(text -> Mono.fromCallable(() -> {
                            acquireSemaphore();
                            return vectorizer.embeddings(text, Optional.empty());
                        })
                        .doOnTerminate(this::releaseSemaphore)
                        .subscribeOn(Schedulers.boundedElastic())
                        .map(this::createEmbeddingsReply)
                        .onErrorResume(e -> {
                            responseObserver.onError(e);
                            return Mono.empty();
                        }))
                .doOnNext(builder::addEmbeddings)
                .then()
                .doOnTerminate(() -> {
                    responseObserver.onNext(builder.build());
                    responseObserver.onCompleted();
                })
                .subscribe();
    }

    private EmbeddingsVectorReply createEmbeddingsReply(float[] embeddings) {
        EmbeddingsVectorReply.Builder builder = EmbeddingsVectorReply.newBuilder();
        for (float value : embeddings) {
            builder.addEmbeddings(value);
        }
        return builder.build();
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

    /**
     * Acquires a permit from the semaphore to limit concurrency.
     */
    private void acquireSemaphore() {
        try {
            semaphore.acquire();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Failed to acquire semaphore", e);
        }
    }

    /**
     * Releases a permit back to the semaphore.
     */
    private void releaseSemaphore() {
        semaphore.release();
    }
}
