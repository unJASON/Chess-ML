#include <iostream>
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"

using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;
int main() {
    const string pathToGraph = "./current_policy.model.meta";
    const string checkpointPath = "./current_policy.model";
    auto session = NewSession(SessionOptions());
    if (session == nullptr)
    {
        throw runtime_error("Could not create Tensorflow session.");
    }

    Status status;

    // 读入我们预先定义好的模型的计算图的拓扑结构
    MetaGraphDef graph_def;
    status = ReadBinaryProto(Env::Default(), pathToGraph, &graph_def);
    if (!status.ok())
    {
        throw runtime_error("Error reading graph definition from " + pathToGraph + ": " + status.ToString());
    }

    // 利用读入的模型的图的拓扑结构构建一个session
    status = session->Create(graph_def.graph_def());
    if (!status.ok())
    {
        throw runtime_error("Error creating graph: " + status.ToString());
    }

    // 读入预先训练好的模型的权重
    Tensor checkpointPathTensor(DT_STRING, TensorShape());
    checkpointPathTensor.scalar<std::string>()() = checkpointPath;
    status = session->Run(
            {{ graph_def.saver_def().filename_tensor_name(), checkpointPathTensor },},
            {},
            {graph_def.saver_def().restore_op_name()},
            nullptr);
    if (!status.ok())
    {
        throw runtime_error("Error loading checkpoint from " + checkpointPath + ": " + status.ToString());
    }

    //dimention: 1*4*15*15,[batchsize,channel,width,height]
    tensorflow::TensorShape shape;
	shape.InsertDim(0,1);
	shape.InsertDim(1,4);
	shape.InsertDim(2,15);
	shape.InsertDim(3,15);
    Tensor input_states(tensorflow::DT_FLOAT, shape);
    //  构造模型的输入，相当与python版本中的feed
    std::vector<std::pair<string, Tensor>> input;
    auto input_states_map = input_states.tensor<float,4>();   //四维向量

    for(int i =0; i< 3;i++){
        for(int j =0; j<15; j++){
            for(int k =0; k<15; k++){
				input_states_map(0,i,j,k) = 0.0f;
            }
        }
    }
	for (int j = 0; j < 15; j++) {
		for (int k = 0; k < 15; k++) {
			input_states_map(0, 3, j, k) = 1.0f;
		}
	}
	input.emplace_back(std::string("Placeholder:0"),input_states);
	//   运行模型，并获取输出
    std::vector<tensorflow::Tensor> answer;
	//dense_2/Tanh:0: win_rate, dense/LogSoftmax:0: probablity of action
    status = session->Run(input, {"dense_2/Tanh:0","dense/LogSoftmax:0"}, {}, &answer);
	cout <<endl<<"answersize:"<< answer.size()<<endl;
    Tensor result = answer[0];
    auto result_map = result.tensor<float,2>();
    cout<<"win_rate: "<<result_map(0,0)<<endl;
    Tensor result2 = answer[1];
	auto result_map2 = result2.tensor<float, 2>();
	for (int i = 0; i < 225; i++) {
		cout << "probablity: " << result_map2(0, i) << endl;
	}
    return 0;

}
