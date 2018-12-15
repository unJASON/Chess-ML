#include <iostream>
#include <dlfcn.h>
#include <ctime>
#include <time.h>
#include<random>
#include<algorithm>
#include <cstring>
#include <fstream>
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include <unistd.h>
#include <windows.h>

using namespace std;
using namespace tensorflow::ops;


//Model���ݲ���
const string pathToGraph = "./current_policy.model.meta";
const string checkpointPath = "./current_policy.model";
//�Ự��������Ϊȫ�ֱ���
tensorflow::Session *session = NULL;
tensorflow::MetaGraphDef graph_def;
//alphaZero����
float c_puct = 5;



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
	float _Q, _U, Prior_p;//alphaZero��Ҫ�õ��Ĳ���
	float win_rate_leaf_temp;	//��ʱ���win_rate

	Node(int c, int t, int x, int y,float prior_p, Node * p = nullptr) {
		parent = p;
		color = c;
		_Q = 0;
		_U = 0;
		Prior_p = prior_p;	//action_Probablity
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


//���򴫲�
void packPropagation(Node*);

//UCT��ʽ(alphaZero)
float evaluate_value(Node *r) {
	r->_U = (c_puct * (r->Prior_p)*sqrt(r->parent->N) / (1 + r->N));
	return r->_Q + r->_U;
};
float update_node_vale(Node *r) {
	r->_Q = r->_Q + 1.0*(r->win_rate_leaf_temp - r->_Q) / r->N;
}
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

//ʹ�������������ǰ��probablity��action
std::vector<tensorflow::Tensor> model_evaluate(Node* n) {
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
	/*��������,����board �뵱ǰ��n,����boardת���õ�*/
	
	/**/

	input.emplace_back(std::string("Placeholder:0"), input_states);
	//   ����ģ�ͣ�����ȡ���
	std::vector<tensorflow::Tensor> answer;
	//dense_2/Tanh:0: win_rate, dense/LogSoftmax:0: probablity of action
	status = session->Run(input, { "dense_2/Tanh:0","dense/LogSoftmax:0" }, {}, &answer);
	return answer;
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
//ģ��
Node * simulate_model(Node* n) {
	++(n->N);
	if (n->isTerminated) {
		if (n->success == -1){
			n->win_rate_leaf_temp = 0;
		}else{
			if (n->color == n->success) {
				n->win_rate_leaf_temp = -1;
			}
			else {
				n->win_rate_leaf_temp = 1;
			}
		}
		
		if (n->success == 1) {
			++(n->QB);
		}
		else if (n->success == 2) {
			++(n->QW);
		}
		return n;
	}
	int opp = n->color;
	int own = 1;
	if (n->color == 1) own = 2;
	//��n��model��evaluate
	std::vector<tensorflow::Tensor> answer = model_evaluate(n);
	n->win_rate_leaf_temp = -answer[0].tensor<float, 2>()(0,0);
	
	return n;
}
Node * select(Node* n) {
	int own = n->color;
	int opp = 1;
	if (own == 1) opp = 2;
	++(n->N);
	//�����վ�ֱ�ӷ���
	if (n->isTerminated) {
		if (n->success == -1) {
			n->win_rate_leaf_temp = 0;
		}
		else {
			if (n->color == n->success) {
				n->win_rate_leaf_temp = -1;
			}
			else {
				n->win_rate_leaf_temp = 1;
			}
		}
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
			double valCh = evaluate_value(u);
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

	}else if (n->newChild) {
		bestChild = n->newChild;
		n->newChild = bestChild->nxt;
		bestChild->nxt = n->first;
		n->first = bestChild;
		//�Ը�Ҷ�ӽڵ���evaluate����ȡʤ���Ŀ�����
		leaf = simulate_model(bestChild);
		//�Ա��ڵ��leaf��win_rate����
		update_node_vale(leaf);
	}
	

	//���򴫲�������Ӯ
	if (leaf->success == 1) {
		++(n->QB);
	}
	else if (leaf->success == 2) {
		++(n->QW);
	}
	//���򴫲�����win_rate,ÿ����һ��ȡ��һ�Σ����±��ڵ��n
	leaf->win_rate_leaf_temp = -leaf->win_rate_leaf_temp;
	n->win_rate_leaf_temp = leaf->win_rate_leaf_temp;
	update_node_vale(n);
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
					Node *v = new Node(c, n->color, i, j, 1, n);
					v->success = v->isTerminated = c;
					v->nxt = nullptr;
					v->hasExpanded = true;
					return v;
				}
			}
		}
	}

	//ʹ��model��evaluate,�����ʣ�����µ������֤success������,ͬʱ���ǵ�����ȫ�����̵Ŀ���
	std::vector<tensorflow::Tensor> answer = model_evaluate(n);
	//����log ����Ҫȥ��
	auto act_probablity = answer[1].tensor<float, 2>();

	//ֻҪû�¹�����չ
	for (int i = 0; i < 15; ++i) {
		for (int j = 0; j < 15; ++j) {
			if (board[i][j]) continue;
			if (true) {
				u = new Node(c, n->color, i,j ,exp(act_probablity(0 ,i*15+j) ), n);
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
	root = new Node(color, turn, x, y, 1, nullptr);
	for (;;) {
		duration = (double)(clock() - timeStart) / CLOCKS_PER_SEC;
		if (duration >= RuningTime) {
			/*�ĳ�������ѡ��*/
			int maxVal = -1;
			
			while (true){
				if (freopen("question.txt", "w", stdout) == NULL) {
					sleep(5);//linux��˯5second
				}else{
					break;
				}
			}
			
			Node * v = nullptr;
			std::cout << root->x << root->y << endl;
			for (Node *u = root->first; u; u = u->nxt) {
				if (maxVal < u->N) {
					maxVal = u->N;
					v = u;
				}
				if (u->nxt){
					std::cout << u->x << "," << u->y << "," << u->N << "|";
				}else{
					std::cout << u->x << "," << u->y << "," << u->N;
				}
				
			}
			fclose(stdout);
			//��Ϣ10S�ȴ�������
			//��answer��8sһ��ֱ���ɹ�Ϊֹ��
			//�鿴�Ƿ�Ϊ�������кŵĴ𰸣���Ϊ���ٵ�10s.�ظ���������
			while (true) {
				if (freopen("ans.txt", "r", stdin) == NULL) {
					sleep(5);//linux��˯5second
				}else {
					break;
				}
			}
			fclose(stdout);
			cin >> v->x >> v->y;

			if (v) {
				cout <<"simulationtimes:"<<v->N << " " << v->x << " " << v->y << " " << endl;
				board[v->x][v->y] = v->color;
				x = v->x;
				y = v->y;
				printBoard();
				return 0;
			}
			else std::cout << "NULL PTR" << endl;
			return -1;
		}
		else {
			//MCTS
			leaf = select(root);
		}
	}
}

//���2--����
int White_Player() {
	double duration = 0;	//����ʱ��
	clock_t timeStart = clock();

	Node * leaf = nullptr;
	int color = 1;
	int turn = 2;
	root = new Node(color, turn, x, y, 1, nullptr);

	for (;;) {
		duration = (double)(clock() - timeStart) / CLOCKS_PER_SEC;
		if (duration >= RuningTime) {
			int maxVal = -1;
			Node * v = nullptr;

			freopen("question.txt", "w", stdout);//д�ļ�����ȡֵ
			std::cout << root->x << root->y << endl;
			Node * v = nullptr;
			for (Node *u = root->first; u; u = u->nxt) {
				if (maxVal < u->N) {
					maxVal = u->N;
					v = u;
				}
				if (u->nxt) {
					std::cout << u->x << "," << u->y << "," << u->N << "|";
				}
				else {
					std::cout << u->x << "," << u->y << "," << u->N;
				}
			}
			fclose(stdout);
			//��Ϣ10S�ȴ�������
			//��answer��8sһ��ֱ���ɹ�Ϊֹ��
			//�鿴�Ƿ�Ϊ�������кŵĴ𰸣���Ϊ���ٵ�10s.�ظ���������
			freopen("ans.txt", "r", stdin);
			cin>>v -> x >>v ->y;
			if (v) {
				cout << "simulationtimes:" << v->N << " " << v->x << " " << v->y << " " << endl;
				board[v->x][v->y] = v->color;
				x = v->x;
				y = v->y;
				printBoard();
				return 0;
			}
			else std::cout << "NULL PTR" << endl;
			return -1;
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
			int result = Black_Player();
			if (result == -1){
				cout << "even" << endl;
			}
			if (judge(board, 1, x, y) == 1)
			{
				cout << "Congratulations to Black_Player" << endl;
				break;
			}
			result = White_Player();
			if (result == -1) {
				cout << "even" << endl;
			}
			if (judge(board, 2, x, y) == 2){
				cout << "Congratulations to White_Player" << endl;
				break;
			}
		}
		//���ý�����ر�TensorFlow�ĻỰ
		closeSessionModel();
	}
}


