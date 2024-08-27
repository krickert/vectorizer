package com.krickert.search.vectorizer.grpc;

import com.krickert.search.service.*;
import com.krickert.search.vectorizer.Vectorizer;
import io.grpc.stub.StreamObserver;
import jakarta.inject.Singleton;

import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;

/**
 * This class represents the implementation of the gRPC endpoint for creating embeddings vectors.
 * It extends the abstract class EmbeddingServiceImplBase, which is the base class for the server implementation
 * of the EmbeddingService.
 */
@Singleton
public class EmbeddingsEndpoint extends EmbeddingServiceGrpc.EmbeddingServiceImplBase {
    private final Vectorizer vectorizer;

    public EmbeddingsEndpoint(Vectorizer vectorizer) {
        this.vectorizer = vectorizer;
    }

    /**
     * Creates an embeddings vector based on the given request and sends the reply to the response observer.
     *
     * @param request           The request containing the text to generate the embeddings vector from.
     * @param responseObserver The response observer to send the reply to.
     */
    @Override
    public void createEmbeddingsVector(EmbeddingsVectorRequest request, StreamObserver<EmbeddingsVectorReply> responseObserver) {
        EmbeddingsVectorReply.Builder builder = EmbeddingsVectorReply.newBuilder();
        Collection<Float> embeddings = vectorizer.getEmbeddings(request.getText());
        builder.addAllEmbeddings(embeddings);
        EmbeddingsVectorReply reply = builder.build();
        sendReply(responseObserver, reply);
    }

    /**
     * Creates embeddings vectors based on the given request and sends the reply to the response observer.
     *
     * @param request           The request containing the text to generate the embeddings vectors from.
     *                           It is an instance of EmbeddingsVectorsRequest.
     * @param responseObserver The response observer to send the reply to.
     *                           It is an instance of StreamObserver<EmbeddingsVectorsReply>.
     *
     * @throws NullPointerException if the request or responseObserver is null.
     */
    @Override
    public void createEmbeddingsVectors(EmbeddingsVectorsRequest request, StreamObserver<EmbeddingsVectorsReply> responseObserver) {
        EmbeddingsVectorsReply.Builder builder = EmbeddingsVectorsReply.newBuilder();
        // Utilizing a parallel stream in Java does not necessarily guarantee that the order of the elements
        // will be maintained. In this case, the order of elements in the output will remain consistent
        // with the order in the input list because the operation we've applied (.map()) is stateless and non-interfering.
        List<EmbeddingsVectorReply> embeddings = request.getTextList().parallelStream()
                .map(text -> {
                    Collection<Float> vector = vectorizer.getEmbeddings(text);
                    EmbeddingsVectorReply.Builder replyBuilder = EmbeddingsVectorReply.newBuilder();
                    replyBuilder.addAllEmbeddings(vector);
                    return replyBuilder.build();
                })
                .collect(Collectors.toList());
        builder.addAllEmbeddings(embeddings);
        sendReply(responseObserver, builder.build());
    }

    @Override
    public void check(HealthCheckRequest request, StreamObserver<HealthCheckReply> responseObserver) {
        HealthCheckReply reply = HealthCheckReply.newBuilder().setStatus("EmbeddingService is running").build();
        sendReply(responseObserver, reply);
    }

    /**
     * Sends the reply to the response observer.
     *
     * @param responseObserver The response observer to send the reply.
     * @param reply            The reply to be sent.
     */
    private <T> void sendReply(StreamObserver<T> responseObserver, T reply) {
        responseObserver.onNext(reply);
        responseObserver.onCompleted();
    }
}
