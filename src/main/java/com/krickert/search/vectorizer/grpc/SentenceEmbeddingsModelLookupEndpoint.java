package com.krickert.search.vectorizer.grpc;

import com.krickert.search.vectorizer.*;
import io.grpc.Status;
import io.grpc.StatusException;
import io.grpc.stub.StreamObserver;
import jakarta.inject.Singleton;

/**
 * This class represents an endpoint for looking up a sentence embedding model based on a request.
 * It extends the abstract class {@link SentenceEmbeddingModelServiceGrpc.SentenceEmbeddingModelServiceImplBase},
 * which serves as the base class for the server implementation of the SentenceEmbeddingModelService.
 * The lookupModel method receives a request and a response observer, and it performs the lookup operation.
 * It converts the model type from the request to an NLPModel, and if found, constructs a reply with the details of the sentence embedding model.
 * If the model type is not found, it throws a NOT_FOUND exception.
 * The convertToNLPModel method converts the SentenceEmbeddingType to an NLPModel enumeration value.
 * The sendReply method is a utility method that sends the reply to the response observer and completes the response.
 */
@Singleton
public class SentenceEmbeddingsModelLookupEndpoint extends SentenceEmbeddingModelServiceGrpc.SentenceEmbeddingModelServiceImplBase {

    /**
     * Looks up a sentence embedding model based on the provided request.
     *
     * @param request           The request for sentence embedding model lookup.
     * @param responseObserver The response observer to send the lookup reply.
     */
    @Override
    public void lookupModel(SentenceEmbeddingModelLookupRequest request, StreamObserver<SentenceEmbeddingModelLookupReply> responseObserver) {
        try {
            SentenceEmbeddingModelLookupReply reply = SentenceEmbeddingModelLookupReply.newBuilder()
                    .setDetails(findSentenceEmbeddingModel(request))
                    .setResponseTime(com.google.protobuf.Timestamp.newBuilder().setSeconds(System.currentTimeMillis() / 1000))
                    .build();
            sendReply(responseObserver, reply);
        } catch (StatusException ex) {
            responseObserver.onError(ex);
        }
    }

    /**
     * Finds the sentence embedding model details based on the provided request.
     *
     * @param request The request for sentence embedding model lookup.
     * @return The sentence embedding model details.
     * @throws StatusException If the model is not found for the given type.
     */
    private SentenceEmbeddings.SentenceEmbeddingModelDetails findSentenceEmbeddingModel(SentenceEmbeddingModelLookupRequest request) throws StatusException {
        NLPModel nlpModel = convertToNLPModel(request.getModelType());

        if (nlpModel != null) {
            return SentenceEmbeddings.SentenceEmbeddingModelDetails.newBuilder()
                    .setModelType(request.getModelType())
                    .setModelName(nlpModel.getModelName())
                    .setUrl(nlpModel.getUrl())
                    .build();
        } else {
            throw Status.NOT_FOUND.withDescription("Model not found for type: " + request.getModelType()).asException();
        }
    }

    /**
     * Converts the given SentenceEmbeddings.SentenceEmbeddingType to the corresponding NLPModel.
     *
     * @param modelType The SentenceEmbeddings.SentenceEmbeddingType to convert.
     * @return The NLPModel corresponding to the given modelType.
     */
    private NLPModel convertToNLPModel(SentenceEmbeddings.SentenceEmbeddingType modelType) {
        switch (modelType) {
            case ALL_MINILM_L12_V2:
                return NLPModel.ALL_MINILM_L12_V2;
            case E5_BASE_v2:
                return NLPModel.E5_BASE_v2;
            case MSMARCO_MINILM_L_6_V3:
                return NLPModel.MSMARCO_MINILM_L_6_V3;
            case MSMARCO_DISTILBERT_BASE_V4:
                return NLPModel.MSMARCO_DISTILBERT_BASE_V4;
            case PARAPHRASE_MULTILINGUAL_MPNET_BASE_V2:
                return NLPModel.PARAPHRASE_MULTILINGUAL_MPNET_BASE_V2;
            default:
                return null;
        }
    }

    /**
     * Sends the given reply to the provided response observer.
     * This method is generic to accept any type of reply.
     *
     * @param <T>              The type of the reply.
     * @param responseObserver The response observer to send the reply.
     * @param reply            The reply to be sent.
     */
    private <T> void sendReply(StreamObserver<T> responseObserver, T reply) {
        responseObserver.onNext(reply);
        responseObserver.onCompleted();
    }
}