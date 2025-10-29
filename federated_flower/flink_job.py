from pyflink.datastream import StreamExecutionEnvironment
from pyflink.common.typeinfo import Types
from pyflink.datastream.connectors import StreamingFileSink
from pyflink.datastream.functions import ProcessFunction, RuntimeContext
from pyflink.datastream.state import ValueStateDescriptor


class MovingAvg(ProcessFunction):
    def open(self, runtime_context: RuntimeContext):
        self.count_state = runtime_context.get_state(ValueStateDescriptor('count', Types.LONG()))
        self.sum_state = runtime_context.get_state(ValueStateDescriptor('sum', Types.FLOAT()))

    def process_element(self, value, ctx: 'ProcessFunction.Context'):
        # value: (timestamp, reward)
        count = self.count_state.value() or 0
        total = self.sum_state.value() or 0.0
        count += 1
        total += float(value[1])
        self.count_state.update(count)
        self.sum_state.update(total)
        avg = total / count
        yield f"{value[0]},{value[1]},{avg:.6f}"


def run_moving_avg(input_path: str, output_path: str):
    env = StreamExecutionEnvironment.get_execution_environment()
    ds = env.read_text_file(input_path) \
         .map(lambda s: s.split(','), output_type=Types.OBJECT_ARRAY(Types.STRING())) \
         .map(lambda arr: (arr[0], float(arr[1])), output_type=Types.TUPLE([Types.STRING(), Types.FLOAT()])) \
         .process(MovingAvg(), output_type=Types.STRING())

    ds.write_as_text(output_path)
    env.execute("moving_avg_bandit_rewards")


if __name__ == "__main__":
    # Example: python federated_flower/flink_job.py input.csv output.txt
    import sys
    run_moving_avg(sys.argv[1], sys.argv[2])


