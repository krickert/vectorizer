package com.krickert.search.service.vectorizer.grpc;

import com.krickert.search.model.pipe.PipeDocument;
import com.krickert.search.model.test.util.TestDataHelper;
import com.krickert.search.service.EmbeddingServiceGrpc;
import com.krickert.search.service.EmbeddingsVectorReply;
import com.krickert.search.service.EmbeddingsVectorRequest;
import com.krickert.search.service.EmbeddingsVectorsReply;
import com.krickert.search.service.EmbeddingsVectorsRequest;
import io.grpc.StatusRuntimeException;
import io.grpc.stub.StreamObserver;
import io.micronaut.core.util.StringUtils;
import io.micronaut.runtime.EmbeddedApplication;
import io.micronaut.test.extensions.junit5.annotation.MicronautTest;
import jakarta.inject.Inject;
import org.apache.commons.compress.utils.Lists;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static java.util.concurrent.TimeUnit.SECONDS;
import static org.awaitility.Awaitility.await;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

@MicronautTest
class GrpcEmbeddingsServiceTest {

    private static final Logger log = LoggerFactory.getLogger(GrpcEmbeddingsServiceTest.class);

    @Inject
    EmbeddedApplication<?> application;

    AtomicInteger finishedDocuments = new AtomicInteger(0);

    @Test
    void testItWorks() {
        Assertions.assertTrue(application.isRunning());
    }

    @Inject
    EmbeddingServiceGrpc.EmbeddingServiceBlockingStub endpoint;
    @Inject
    EmbeddingServiceGrpc.EmbeddingServiceStub endpoint2;

    private final Collection<EmbeddingsVectorReply> finishedEmbeddingsVectorReply = Lists.newArrayList();
    StreamObserver<EmbeddingsVectorReply> streamEmbeddingsVectorReplyObserver = new StreamObserver<>() {
        @Override
        public void onNext(EmbeddingsVectorReply reply) {
            log.info("Received {} embeddings vector reply of size: {}", finishedDocuments.getAndIncrement(), reply.getEmbeddingsCount());
            finishedEmbeddingsVectorReply.add(reply);
        }

        @Override
        public void onError(Throwable throwable) {
            log.error("Not implemented", throwable);
        }

        @Override
        public void onCompleted() {
            log.debug("Finished 1");
        }
    };

    private final Collection<EmbeddingsVectorsReply> finishedEmbeddingsVectorsReply = Lists.newArrayList();
    StreamObserver<EmbeddingsVectorsReply> streamEmbeddingsVectorsReplyObserver = new StreamObserver<>() {
        AtomicInteger counter = new AtomicInteger(0);
        @Override
        public void onNext(EmbeddingsVectorsReply reply) {
            finishedEmbeddingsVectorsReply.add(reply);
            counter.incrementAndGet();
            if (counter.get() % 10 == 0) {
                log.info("number of docs processed {}", counter.get());
            }
        }

        @Override
        public void onError(Throwable throwable) {
            log.error("Not implemented", throwable);
        }

        @Override
        public void onCompleted() {
            log.info("Finished 2");
        }
    };

    @Test
    void testEmbeddingsVectorServerEndpoint() {
        AtomicInteger finishedDocuments = new AtomicInteger(0);
        try {
            // get the bodies of the documents in form of a list
            List<String> documentBodies = TestDataHelper.getFewHunderedPipeDocuments().stream().map(PipeDocument::getBody).toList();

            // process the document bodies in parallel
            documentBodies.parallelStream().forEach(text -> {
                final String textToSend;
                if (StringUtils.isEmpty(text)) {
                    log.warn("Empty text for test!!!  Replacing with dummy");
                    textToSend = "Empty Body";
                } else {
                    textToSend = text;
                }
                EmbeddingsVectorRequest request = EmbeddingsVectorRequest.newBuilder()
                        .setText(textToSend).build();
                EmbeddingsVectorReply reply;
                try {
                    reply = endpoint.createEmbeddingsVector(request);
                    assertNotNull(reply);
                    assertTrue(reply.getEmbeddingsList().size() > 100);
                    finishedDocuments.incrementAndGet();
                } catch (StatusRuntimeException e) {
                    log.error("Last embedding throws an exception. Text: [{}] Finished Docs: [{}] Exception: [{}]", text,
                            finishedDocuments.get(), e.getMessage());
                    throw e;
                }
            });

        } catch (Exception e) {
            log.error("Error occurred while creating embedding vector: ", e);
            throw new RuntimeException(e);
        }
    }

    @Test
    void testEmbeddingsVectorAsyncEndpoint() {
        Collection<String> titles = TestDataHelper.getFewHunderedPipeDocuments().stream().map(PipeDocument::getTitle).toList();
        for (String title : titles) {
            EmbeddingsVectorRequest request = EmbeddingsVectorRequest.newBuilder()
                    .setText(title).build();
            endpoint2.createEmbeddingsVector(request, streamEmbeddingsVectorReplyObserver);

        }
        log.info("waiting up to 15 seconds for at least 1 document to be added..");
        await().atMost(25, SECONDS).until(() -> finishedEmbeddingsVectorReply.size() > 1);
        log.info("waiting up to 25 seconds for at least 10 docs to be added..");
        await().atMost(30, SECONDS).until(() -> finishedEmbeddingsVectorReply.size() > 10);
        log.info("waiting for 500 seconds max for all 367 documents to be processed..");
        await().atMost(500, SECONDS).until(() -> finishedEmbeddingsVectorReply.size() == 367);
    }

    @Test
    void testEmbeddingsVectorsFromBodyParagraphsSync() {
        AtomicInteger counter = new AtomicInteger(0);
        try {
            // get the body paragraphs of the documents in form of a list
            Collection<PipeDocument> pipeDocuments = TestDataHelper.getFewHunderedPipeDocuments();
            List<List<String>> documentParagraphs  = Lists.newArrayList();
            for (PipeDocument doc : pipeDocuments) {
                documentParagraphs.add(doc.getBodyParagraphsList().stream().toList());
            }

            // process the document paragraphs in parallel
            documentParagraphs.forEach(paragraphs -> {
                EmbeddingsVectorsRequest request = EmbeddingsVectorsRequest.newBuilder()
                        .addAllText(paragraphs)
                        .build();
                EmbeddingsVectorsReply reply;
                try {
                    reply = endpoint.createEmbeddingsVectors(request);
                    assertNotNull(reply);
                    if (reply.getEmbeddingsCount() == 0) {
                        log.info("Paragraphs returned 0: " + paragraphs);

                    }
                    //assertTrue(reply.getEmbeddingsCount() > 0);
                    counter.incrementAndGet();
                    if (counter.get() % 10 == 0) {
                        log.info("number of docs processed {}", counter.get());
                    }
                } catch (StatusRuntimeException e) {
                    log.error("Error occurred while creating embedding vectors for paragraphs: Finished Docs: [{}] Exception: [{}]", counter.get(), e.getMessage());
                    throw e;
                }
            });

        } catch (Exception e) {
            log.error("Error occurred while creating embedding vectors from paragraphs: ", e);
            throw new RuntimeException(e);
        }
    }

    @Test
    void testEmbeddingsVectorsFromBodyParagraphsAsync() {
        Collection<PipeDocument> pipeDocuments = TestDataHelper.getFewHunderedPipeDocuments();
        List<List<String>> documentParagraphs  = Lists.newArrayList();
        for (PipeDocument doc : pipeDocuments) {
            documentParagraphs.add(doc.getBodyParagraphsList().stream().toList());
        }

        for (List<String> paragraphs : documentParagraphs) {
            EmbeddingsVectorsRequest request = EmbeddingsVectorsRequest.newBuilder()
                    .addAllText(paragraphs)
                    .build();
            endpoint2.createEmbeddingsVectors(request, streamEmbeddingsVectorsReplyObserver);
        }
        log.info("waiting up to 15 seconds for at least 1 document to be added..");
        await().atMost(25, SECONDS).until(() -> finishedEmbeddingsVectorsReply.size() > 1);
        log.info("waiting up to 25 seconds for at least 10 docs to be added..");
        await().atMost(30, SECONDS).until(() -> finishedEmbeddingsVectorsReply.size() > 10);
        log.info("waiting for 500 seconds max for all 367 documents to be processed..");
        await().atMost(500, SECONDS).until(() -> finishedEmbeddingsVectorsReply.size() == 367);
    }
}
