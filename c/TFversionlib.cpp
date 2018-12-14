#include <iostream>
#include <ctime>
#include <time.h>
#include<random>
#include<algorithm>
#include <cstring>
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"

using namespace std;
using namespace tensorflow::ops;
int board[15][15];
int sec = 1000;
const double RuningTime = 0.95f;
float confident = 1.96;
int equivalence = 1000;
int x = -1;//最后一步下的地方
int y = -1;//最后一步下的地方

//扩展的点数
const int expand_CNT = 225;
//评估函数初始权值
const int INITWEIGHT = 10000;
const double C = 1.0f;

//黑色:1 自己; 白色: 2,对手
//节点数据结构
struct Node {
	int N, QB, QW;	//模拟次数,黑赢次数,白赢次数
	Node *first, *nxt;
	Node * newChild;
	Node * parent;
	int color;	//颜色，最后一步所下的颜色
	int x;	//下棋点
	int y;	//下棋点
	//unsigned int visited[15][15];
	int depth;	//探索深度
	int isTerminated;	//是否为终局
	int turn;	//当前应当下棋的颜色
	bool hasExpanded;	//是否已扩展
	unsigned int success;
	long levelScore;	//层级评分
	long score;		//自身评分

	Node(int c, int t, int x, int y, long scr, long level_Score, Node * p = nullptr) {
		parent = p;
		color = c;

		levelScore = level_Score;
		score = scr;

		this->x = x;
		this->y = y;
		first = NULL;
		nxt = NULL;
		newChild = NULL;
		N = QB = QW = 0;
		if (p) depth = p->depth + 1;
		else depth = 0;
		isTerminated = false;
		hasExpanded = false;
		turn = t;
		success = 0;
	}
};
//判断输赢需要用到的数组
int fx[4] = { 0, -1, -1, -1 };
int fy[4] = { -1, 0, -1, 1 };
int tx[4] = { 0, 1, 1, 1 };
int ty[4] = { 1, 0, 1, -1 };
Node* root;	//root 指针
   //扩展
Node* expand(Node*);
//模拟
Node* simulate(Node*);

//反向传播
void packPropagation(Node*);

//UCT公式
double evaluate(Node *r);
//打印board
void printBoard();
//判断输赢，如果赢则返回颜色，如果未赢则返回0
int judge(int(*)[15], int, int, int);




//判断输赢，如果赢则返回颜色，如果未赢则返回0
int judge(int(*board)[15], int color, int x, int y) {
	int chains = 0;
	int nx;
	int ny;

	for (int i = 0; i < 4; ++i) {
		nx = x;
		ny = y;
		chains = 0;
		for (int j = 0; j < 4; ++j) {
			nx += fx[i];
			ny += fy[i];
			if (nx < 0 || nx>14 || ny < 0 || ny>14) break;
			if (board[nx][ny] == color) {
				++chains;
				if (chains >= 4) {
					return color;
				}
			}
			else break;
		}
		nx = x;
		ny = y;

		for (int j = 0; j < 4; ++j) {
			nx += tx[i];
			ny += ty[i];

			if (nx < 0 || nx>14 || ny < 0 || ny>14) break;
			if (board[nx][ny] == color) {
				++chains;
				if (chains >= 4) {
					return color;
				}
			}
			else break;
		}
	}
	return 0;
}
//扩展 考虑扩展的时候返回的是否为终局节点
Node * expand(Node* n) {
	int c = 1;
	if (n->color == 1) c = 2;
	Node * u = euldVis(n);
	if (u->isTerminated) {
		n->newChild = u;
		return n;
	}
	n->newChild = u;
	//n->newChild = scoreBoard(n);

	return n;
}



//Model数据部分
const string pathToGraph = "./current_policy.model.meta";
const string checkpointPath = "./current_policy.model";
//会话，数据流为全局变量
tensorflow::Session *session = NULL;
tensorflow::MetaGraphDef graph_def;
//alphaZero参数
float c_puct = 5;


//使用神经网络分析当前的probablity与action
void evaluate(/*board*/) {
	// 读入预先训练好的模型的权重
	tensorflow::Tensor checkpointPathTensor(tensorflow::DT_STRING, tensorflow::TensorShape());
	checkpointPathTensor.scalar<std::string>()() = checkpointPath;

	tensorflow::Status status;
	status = session->Run(
		{ { graph_def.saver_def().filename_tensor_name(), checkpointPathTensor }, },
		{},
		{ graph_def.saver_def().restore_op_name() },
		nullptr);
	if (!status.ok())
	{
		throw runtime_error("Error loading checkpoint from " + checkpointPath + ": " + status.ToString());
	}

	//dimention: 1*4*15*15,[batchsize,channel,width,height]
	tensorflow::Tensor input_states(tensorflow::DT_FLOAT, tensorflow::TensorShape({ 1, 4, 15, 15}));
	//  构造模型的输入，相当与python版本中的feed
	std::vector<std::pair<string, tensorflow::Tensor>> input;
	auto input_states_map = input_states.tensor<float, 4>();   //四维向量
	/*输入数据*/
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 15; j++) {
			for (int k = 0; k < 15; k++) {
				input_states_map(0, i, j, k) = 0.0f;
			}
		}
	}
	for (int j = 0; j < 15; j++) {
		for (int k = 0; k < 15; k++) {
			input_states_map(0, 3, j, k) = 1.0f;
		}
	}
	/*根据board转换得到*/

	input.emplace_back(std::string("Placeholder:0"), input_states);
	//   运行模型，并获取输出
	std::vector<tensorflow::Tensor> answer;
	//dense_2/Tanh:0: win_rate, dense/LogSoftmax:0: probablity of action
	status = session->Run(input, { "dense_2/Tanh:0","dense/LogSoftmax:0" }, {}, &answer);
	/*cout << endl << "answersize:" << answer.size() << endl;
	Tensor result = answer[0];
	auto result_map = result.tensor<float, 2>();
	cout << "win_rate: " << result_map(0, 0) << endl;
	Tensor result2 = answer[1];
	auto result_map2 = result2.tensor<float, 2>();
	for (int i = 0; i < 225; i++) {
		cout << "probablity: " << result_map2(0, i) << endl;
	}*/
}
void initSessionModel() {
	//初始化session
	session = tensorflow::NewSession(tensorflow::SessionOptions());
	if (session == nullptr)
	{
		throw runtime_error("Could not create Tensorflow session.");
	}

	// 读入我们预先定义好的模型的计算图的拓扑结构
	tensorflow::Status status = tensorflow::ReadBinaryProto(tensorflow::Env::Default(), pathToGraph, &graph_def);
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

}

Node * select(Node* n) {
	int own = n->color;
	int opp = 1;
	if (own == 1) opp = 2;
	++(n->N);
	//出现终局直接返回
	if (n->isTerminated) {
		if (n->success == 1) ++(n->QB);
		else if (n->success == 2) ++(n->QW);
		return n;
	}
	if (!n->hasExpanded) {
		expand(n);
		n->hasExpanded = true;
		//当扩展到最后一个节点后，应当注意此时棋盘上所有点都被占据了！！！再也没有child了。。
		if (n->newChild == NULL) {
			n->isTerminated = true;
			n->success = -1;
			return n;
		}
	}
	Node * leaf = nullptr;

	Node * bestChild = NULL;

	if (n->newChild == NULL) {
		double minVal = -10e3;
		for (Node * u = n->first; u; u = u->nxt) {
			double valCh = evaluate(u);
			if (valCh > minVal) {
				bestChild = u;
				minVal = valCh;
			}
		}

		if (bestChild) {
			int x = bestChild->x;
			int y = bestChild->y;
			board[x][y] = bestChild->color;
			leaf = select(bestChild);
			board[x][y] = 0;
		}

	}
	else if (n->newChild) {
		bestChild = n->newChild;
		n->newChild = bestChild->nxt;
		bestChild->nxt = n->first;
		n->first = bestChild;

		leaf = simulate(bestChild);

	}

	//反向传播带回输赢
	if (leaf->success == 1) {
		++(n->QB);
	}
	else if (leaf->success == 2) {
		++(n->QW);
	}
	return leaf;
}
Node * euldVis(Node * n) {

	int c = 1;
	if (n->color == 1) c = 2;
	int  cnt = 0;
	Node *u = nullptr;
	Node *v = nullptr;
	int dis1 = -1, dis2 = -1;

	
	for (int i = 0; i < 15; ++i) {
		for (int j = 0; j < 15; ++j) {
			if (board[i][j]) continue;
			if (true) {
				//先判断终局
				if (judge(board, c, i, j) == c) {
					Node *v = new Node(c, n->color, i, j, 1, 1, n);
					v->success = v->isTerminated = c;
					v->nxt = nullptr;
					v->hasExpanded = true;
					return v;
				}
			}
		}
	}

	//只要没下过就扩展
	for (int i = 0; i < 15; ++i) {
		for (int j = 0; j < 15; ++j) {
			if (board[i][j]) continue;
			if (true) {
				u = new Node(c, n->color, i,j , ？,？, n);
				u->nxt = v;
				v = u;
			}
		}
	}
	return u;
}

//扩展 考虑扩展的时候返回的是否为终局节点
Node * expand(Node* n) {
	int c = 1;
	if (n->color == 1) c = 2;
	Node * u = euldVis(n);
	if (u->isTerminated) {
		n->newChild = u;
		return n;
	}
	n->newChild = u;
	//n->newChild = scoreBoard(n);

	return n;
}
//打印棋盘状态
void printBoard() {
	//棋盘状态
	for (int i = 0; i < 15; i++) {
		for (int j = 0; j < 15; j++) {
			if (board[j][i] == 0)
				std::cout << "0 ";
			else if (board[j][i] == 1)
				std::cout << "* ";
			else std::cout << "# ";
		}
		std::cout << '\n';
	}
	//cin.get();
}

void closeSessionModel() {
	//关闭会话
	session->Close();
	//释放session指针
	free(session);
};

//玩家1--黑子
int Black_Player() {
	//memset(board, 0, sizeof(board));
	double duration = 0;	//运行时间
	clock_t timeStart = clock();
	Node * leaf = nullptr;
	int color = 2;
	int turn = 1;
	root = new Node(color, turn, x, y, -1, -1, nullptr);
	for (;;) {
		duration = (double)(clock() - timeStart) / CLOCKS_PER_SEC;
		if (duration >= RuningTime) {
			int maxVal = -1;
			Node * v = nullptr;
			for (Node *u = root->first; u; u = u->nxt) {
				if (maxVal < u->N) {
					maxVal = u->N;
					v = u;
				}
			}

			if (v) {
				cout << v->x << " " << v->y << " " << endl;
				board[v->x][v->y] = v->color;
				x = v->x;
				y = v->y;
				printBoard();
			}
			else std::cout << "NULL PTR" << endl;
			return 0;
		}
		else {
			//MCTS
			leaf = select(root);
		}
	}
}

//玩家2--白子
int White_Player() {
	//memset(board, 0, sizeof(board));
	double duration = 0;	//运行时间
	clock_t timeStart = clock();

	Node * leaf = nullptr;
	int color = 1;
	int turn = 2;
	root = new Node(color, turn, x, y, -1, -1, nullptr);

	for (;;) {
		duration = (double)(clock() - timeStart) / CLOCKS_PER_SEC;
		if (duration >= RuningTime) {
			int maxVal = -1;
			Node * v = nullptr;
			for (Node *u = root->first; u; u = u->nxt) {
				if (maxVal < u->N) {
					maxVal = u->N;
					v = u;
				}
			}

			if (v) {
				cout << v->x << " " << v->y << " " << endl;
				board[v->x][v->y] = v->color;
				x = v->x;
				y = v->y;
				printBoard();
			}
			else std::cout << "NULL PTR" << endl;
			return 0;
		}
		else {
			//MCTS
			leaf = select(root);
		}
	}
}





extern "C"{
	void startSelfPlay() {
		//初始化session并加载模型以后不用每次计算都初始化
		initSessionModel();
		
		//蒙特卡洛
		while (1)
		{
			Black_Player();
			if (judge(board, 1, x, y) == 1)
			{
				cout << "Congratulations to Black_Player！！！" << endl;
				break;
			}
			White_Player();
			if (judge(board, 2, x, y) == 2)
			{
				cout << "Congratulations to White_Player！！！" << endl;
				break;
			}
		}

		
		//调用结束后关闭TensorFlow的会话
		closeSessionModel();
	}
}


