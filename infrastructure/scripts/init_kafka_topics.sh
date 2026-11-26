#!/usr/bin/env bash
set -e

BOOTSTRAP_SERVER="${1:-kafka:9092}"
PARTITIONS=4
REPLICATION_FACTOR=1

echo "Waiting for Kafka broker at ${BOOTSTRAP_SERVER} to become ready..."

# Wait up to 60 seconds for Kafka to be reachable
MAX_RETRIES=30
RETRY_COUNT=0

until /opt/kafka/bin/kafka-topics.sh --bootstrap-server "${BOOTSTRAP_SERVER}" --list > /dev/null 2>&1; do
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ ${RETRY_COUNT} -ge ${MAX_RETRIES} ]; then
        echo "Error: Timed out waiting for Kafka at ${BOOTSTRAP_SERVER}."
        exit 1
    fi
    echo "Kafka not ready yet (attempt ${RETRY_COUNT}/${MAX_RETRIES}). Retrying in 2 seconds..."
    sleep 2
done

echo "Kafka is ready! Creating application topics with ${PARTITIONS} partitions..."

TOPICS=("raw-telemetry" "predictions-log" "drift-alerts")

for TOPIC in "${TOPICS[@]}"; do
    echo "Creating topic: ${TOPIC}..."
    /opt/kafka/bin/kafka-topics.sh \
        --bootstrap-server "${BOOTSTRAP_SERVER}" \
        --create \
        --if-not-exists \
        --topic "${TOPIC}" \
        --partitions "${PARTITIONS}" \
        --replication-factor "${REPLICATION_FACTOR}"
done

echo "Successfully verified Kafka topics:"
/opt/kafka/bin/kafka-topics.sh --bootstrap-server "${BOOTSTRAP_SERVER}" --list

echo "Kafka topic initialization completed successfully."
