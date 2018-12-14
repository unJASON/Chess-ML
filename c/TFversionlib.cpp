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
int x = -1;//���һ���µĵط�
int y = -1;//���һ���µĵط�

//��չ�ĵ���
const int expand_CNT = 225;
//����������ʼȨֵ
const int INITWEIGHT = 10000;
const double C = 1.0f;

//��ɫ:1 �Լ�; ��ɫ: 2,����
//�ڵ����ݽṹ
struct Node {
	int N, QB, QW;	//ģ�����,��Ӯ����,��Ӯ����
	Node *first, *nxt;
	Node * newChild;
	Node * parent;
	int color;	//��ɫ�����һ�����µ���ɫ
	int x;	//�����
	int y;	//�����
	//unsigned int visited[15][15];
	int depth;	//̽�����
	int isTerminated;	//�Ƿ�Ϊ�վ�
	int turn;	//��ǰӦ���������ɫ
	bool hasExpanded;	//�Ƿ�����չ
	unsigned int success;
	long levelScore;	//�㼶����
	long score;		//��������

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
//�ж���Ӯ��Ҫ�õ�������
int fx[4] = { 0, -1, -1, -1 };
int fy[4] = { -1, 0, -1, 1 };
int tx[4] = { 0, 1, 1, 1 };
int ty[4] = { 1, 0, 1, -1 };
Node* root;	//root ָ��
   //��չ
Node* expand(Node*);
//ģ��
Node* simulate(Node*);

//���򴫲�
void packPropagation(Node*);

//UCT��ʽ
double evaluate(Node *r);
//��ӡboard
void printBoard();
//�ж���Ӯ�����Ӯ�򷵻���ɫ�����δӮ�򷵻�0
int judge(int(*)[15], int, int, int);




//�ж���Ӯ�����Ӯ�򷵻���ɫ�����δӮ�򷵻�0
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
//��չ ������չ��ʱ�򷵻ص��Ƿ�Ϊ�վֽڵ�
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



//Model���ݲ���
const string pathToGraph = "./current_policy.model.meta";
const string checkpointPath = "./current_policy.model";
//�Ự��������Ϊȫ�ֱ���
tensorflow::Session *session = NULL;
tensorflow::MetaGraphDef graph_def;
//alphaZero����
float c_puct = 5;


//ʹ�������������ǰ��probablity��action
void evaluate(/*board*/) {
	// ����Ԥ��ѵ���õ�ģ�͵�Ȩ��
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
	//  ����ģ�͵����룬�൱��python�汾�е�feed
	std::vector<std::pair<string, tensorflow::Tensor>> input;
	auto input_states_map = input_states.tensor<float, 4>();   //��ά����
	/*��������*/
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
	/*����boardת���õ�*/

	input.emplace_back(std::string("Placeholder:0"), input_states);
	//   ����ģ�ͣ�����ȡ���
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
	//��ʼ��session
	session = tensorflow::NewSession(tensorflow::SessionOptions());
	if (session == nullptr)
	{
		throw runtime_error("Could not create Tensorflow session.");
	}

	// ��������Ԥ�ȶ���õ�ģ�͵ļ���ͼ�����˽ṹ
	tensorflow::Status status = tensorflow::ReadBinaryProto(tensorflow::Env::Default(), pathToGraph, &graph_def);
	if (!status.ok())
	{
		throw runtime_error("Error reading graph definition from " + pathToGraph + ": " + status.ToString());
	}

	// ���ö����ģ�͵�ͼ�����˽ṹ����һ��session
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
	//�����վ�ֱ�ӷ���
	if (n->isTerminated) {
		if (n->success == 1) ++(n->QB);
		else if (n->success == 2) ++(n->QW);
		return n;
	}
	if (!n->hasExpanded) {
		expand(n);
		n->hasExpanded = true;
		//����չ�����һ���ڵ��Ӧ��ע���ʱ���������е㶼��ռ���ˣ�������Ҳû��child�ˡ���
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

	//���򴫲�������Ӯ
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
				//���ж��վ�
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

	//ֻҪû�¹�����չ
	for (int i = 0; i < 15; ++i) {
		for (int j = 0; j < 15; ++j) {
			if (board[i][j]) continue;
			if (true) {
				u = new Node(c, n->color, i,j , ��,��, n);
				u->nxt = v;
				v = u;
			}
		}
	}
	return u;
}

//��չ ������չ��ʱ�򷵻ص��Ƿ�Ϊ�վֽڵ�
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
//��ӡ����״̬
void printBoard() {
	//����״̬
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
	//�رջỰ
	session->Close();
	//�ͷ�sessionָ��
	free(session);
};

//���1--����
int Black_Player() {
	//memset(board, 0, sizeof(board));
	double duration = 0;	//����ʱ��
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

//���2--����
int White_Player() {
	//memset(board, 0, sizeof(board));
	double duration = 0;	//����ʱ��
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
		//��ʼ��session������ģ���Ժ���ÿ�μ��㶼��ʼ��
		initSessionModel();
		
		//���ؿ���
		while (1)
		{
			Black_Player();
			if (judge(board, 1, x, y) == 1)
			{
				cout << "Congratulations to Black_Player������" << endl;
				break;
			}
			White_Player();
			if (judge(board, 2, x, y) == 2)
			{
				cout << "Congratulations to White_Player������" << endl;
				break;
			}
		}

		
		//���ý�����ر�TensorFlow�ĻỰ
		closeSessionModel();
	}
}


