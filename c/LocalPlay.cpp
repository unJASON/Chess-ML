#include <iostream>
#include <ctime>
#include <time.h>
#include<random>
#include<algorithm>
#include <cstring>
using namespace std;
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
//默认已全是0
    int board[15][15];
    int sec = 1000;
    const double RuningTime = 0.95f;
    float confident = 1.96;
    int equivalence = 1000;
    int x = -1;//最后一步下的地方
    int y = -1;//最后一步下的地方

    //扩展的点数
    const int expand_CNT = 15;
    //评估函数初始权值
    const int INITWEIGHT = 10000;

    const double C = 1.0f;

    //产生随机数
    default_random_engine engine(time(nullptr));
    uniform_int_distribution<int> uidis(0, 224);

    //引入score用于选点
    struct Point {
        int x;
        int y;
        int score;
    };
    bool cmpScore(const Point & a, const Point & b) {
        return a.score > b.score;
    }
    //存储score
    Point pointScore[256];

    static unsigned long x_random = 123456789, y_random = 362436069, z_random = 521288629;
    inline unsigned long xorshf96(void) {          //period 2^96-1
        unsigned long t;
        x_random ^= x_random << 16;
        x_random ^= x_random >> 5;
        x_random ^= x_random << 1;

        t = x_random;
        x_random = y_random;
        y_random = z_random;
        z_random = t ^ x_random ^ y_random;

        return z_random;
    }

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

    //UCT 公式 + 静态评估函数
    double evaluate(Node *r) {
        //if (r->isTerminated) return r->isTerminated;
        if (r->N == 0) return -1;
        if (r->parent == NULL) return -1;


        double s1;
        if (r->color == 1)
            s1 = (double)(r->QB) / (double)(r->N);
        else {
            s1 = (double)(r->QW) / (double)(r->N);
        }

        double s2 = (double)(log((double)(r->parent->N))) / (double)(r->N);
        s2 = sqrt(s2)*C;

        //静态评估函数
        double s3 = (double)(r->score) / double(r->levelScore);
        double weight = 1 / (double)((r->N) + 1);

        //取权值，模拟次数越多则静态评估函数影响越小
        return (s1 + s2)* (1 - weight) + s3 * weight;
    }
    //评估局面返回最好的前N个点。
    Node * scoreBoard(Node * r) {
        return NULL;
    }
    void judgeScore(int(*board)[15], int color, int x, int y);
    const int SIZE = 15;
    //返回评估后的点(需要通过评估函数进行剪枝运算)
    Node * euldVis(Node * n) {

        int c = 1;
        if (n->color == 1) c = 2;
        int  cnt = 0;
        Node *u = nullptr;
        Node *v = nullptr;
        int dis1 = -1, dis2 = -1;

        for (int i = 0; i < SIZE; ++i) {
            for (int j = 0; j < SIZE; ++j) {
                pointScore[i*SIZE + j].x = i;
                pointScore[i*SIZE + j].y = j;
                pointScore[i*SIZE + j].score = -INITWEIGHT * 10;
            }
        }
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

        //给各个点打分
        for (int i = 0; i < 15; ++i) {
            for (int j = 0; j < 15; ++j) {
                if (board[i][j]) continue;
                if (true) {
                    //算分
                    judgeScore(board, c, i, j);
                }
            }
        }
    //#ifdef DEBUG
    //	//cout <<"4,9:"<<pointScore[4 * 15 + 9].score<<endl;
    //	//cout <<"10,9:"<<pointScore[10 * 15 + 9].score<<endl;
    //#endif
        sort(pointScore, pointScore + 15 * 15, cmpScore);

        //静态评估函数估分需要使用
        double levelScore = 0;
        for (int i = 0; i < expand_CNT; ++i) {
            if (!board[pointScore[i].x][pointScore[i].y]) {
                levelScore = levelScore + pointScore[i].score;
            }
        }
        int final_exp = expand_CNT;

        while (true) {
            if (pointScore[final_exp].score >= 222000) {
                levelScore = levelScore + pointScore[final_exp].score;
                final_exp++;
            }
            else {
                break;
            }
        }
        for (int i = 0; i < expand_CNT; ++i) {
            //可能棋盘最后没有那么多点了，所以要考虑到
            if (!board[pointScore[i].x][pointScore[i].y]) {

                u = new Node(c, n->color, pointScore[i].x, pointScore[i].y, pointScore[i].score, levelScore, n);
    #ifdef DEBUG
                //cout <<"x:"<< pointScore[i].x<<"y:"<< pointScore[i].y <<"score:" << pointScore[i].score<<endl;
    #endif
                u->nxt = v;
                v = u;
            }
        }

        //加上可能活三的点
        final_exp = expand_CNT;
        while (true){
            if (pointScore[final_exp].score >= 222000){
                u = new Node(c, n->color, pointScore[final_exp].x, pointScore[final_exp].y, pointScore[final_exp].score, levelScore, n);
                u->nxt = v;
                v = u;
                final_exp++;
            }
            else{
                break;
            }
        }
        return u;
    }

    //color 我的颜色
    void judgeScore(int(*board)[15], int color, int x, int y) {
        //printBoard();
        int opp = 1;
        if (color == 1) opp = 2;
        int blankNum;	//连续空白不能太多
        int nx;
        int ny;
        int total_oppScore = 0;
        int total_selfScore = 0;
        int oppScore = INITWEIGHT;
        int selfScore = INITWEIGHT;
        int weight;	//记录联乘个数
        //记录4个对角的方向算score并求和
        for (int i = 0; i < 4; ++i) {
            //计算自己的分数
            nx = x;
            ny = y;
            blankNum = 0;
            weight = INITWEIGHT;
            for (int j = 0; j < 4; ++j) {
                nx += fx[i];
                ny += fy[i];
                //遇到边界停下并减分
                if (nx < 0 || nx>14 || ny < 0 || ny>14) {
                    blankNum = 0;
                    selfScore = selfScore / 10;
                    break;
                }
                if (board[nx][ny] == color) {//遇到自己的颜色
                    //遇到相同颜色则求和
                    blankNum = 0;
                    selfScore = selfScore + weight * 10;
                    weight = weight * 10;
                }
                else if (board[nx][ny] == opp) {//遇到对手的颜色
                    //停下并减分
                    blankNum = 0;
                    selfScore = selfScore / 10;
                    break;
                }
                else {	//没有棋子
                    if (blankNum > 1) {
                        break;
                    }
                    else {
                        //连续空白最好不要太多
                        blankNum++;
                        weight = weight / 10;
                        continue;
                    }

                }
            }
            //另外一个方向重新计数
            nx = x;
            ny = y;
            blankNum = 0;
            for (int j = 0; j < 4; ++j) {
                nx += tx[i];
                ny += ty[i];
                //遇到边界停下并减分
                if (nx < 0 || nx>14 || ny < 0 || ny>14) {
                    blankNum = 0;
                    selfScore = selfScore / 10;
                    break;
                }
                if (board[nx][ny] == color) {
                    //遇到相同颜色则求和
                    blankNum = 0;
                    selfScore = selfScore + weight * 10;
                    weight = weight * 10;
                }
                else if (board[nx][ny] == opp) {//遇到对手的颜色
                    //停下并减分
                    blankNum = 0;
                    selfScore = selfScore / 10;
                    break;
                }
                else {
                    if (blankNum > 1) {
                        break;
                    }
                    else {
                        //连续空白最好不要太多
                        blankNum++;
                        weight = weight / 10;
                        continue;
                    }
                }
            }

            //计算对手的分数
            nx = x;
            ny = y;
            blankNum = 0;
            weight = INITWEIGHT;
            for (int j = 0; j < 4; ++j) {
                nx += fx[i];
                ny += fy[i];
                //遇到边界停下并减分
                if (nx < 0 || nx>14 || ny < 0 || ny>14) {
                    blankNum = 0;
                    oppScore = oppScore / 10;
                    break;
                }
                if (board[nx][ny] == opp) {//遇到对手的颜色
                    //遇到相同颜色则求和
                    blankNum = 0;
                    oppScore = oppScore + weight * 10;
                    weight = weight * 10;
                }
                else if (board[nx][ny] == color) {//遇到自己的颜色
                    //停下并减分
                    blankNum = 0;
                    oppScore = oppScore / 10;
                    break;
                }
                else {	//没有棋子
                    if (blankNum > 1) {
                        break;
                    }
                    else {
                        //连续空白最好不要太多
                        blankNum++;
                        weight = weight / 10;
                        continue;
                    }

                }
            }
            nx = x;
            ny = y;
            //另外一个方向重新计数
            blankNum = 0;
            for (int j = 0; j < 4; ++j) {
                nx += tx[i];
                ny += ty[i];
                //遇到边界停下并减分
                if (nx < 0 || nx>14 || ny < 0 || ny>14) {
                    blankNum = 0;
                    oppScore = oppScore / 10;
                    break;
                }
                if (board[nx][ny] == opp) {
                    //遇到相同颜色则求和
                    blankNum = 0;
                    oppScore = oppScore + weight * 10;
                    weight = weight * 10;
                }
                else if (board[nx][ny] == color) {//遇到自己的颜色减分跳出
                    blankNum = 0;
                    oppScore = oppScore / 10;
                    break;
                }
                else {
                    if (blankNum > 1) {//空白太多则跳出
                        break;
                    }
                    else {
                        //连续空白最好不要太多
                        blankNum++;
                        weight = weight / 10;
                        continue;
                    }
                }
            }

            total_oppScore = total_oppScore + oppScore;
            total_selfScore = total_selfScore + selfScore;
        }
        //pointScore[x*SIZE + y].score = selfScore > oppScore ? selfScore : oppScore;
        pointScore[x*SIZE + y].score = total_oppScore + total_selfScore;
    }

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

    //随机序列生成
    int* buildRandomSequence(int length) {
        int* array = new int[length];
        for (int i = 0; i < length; i++) {
            array[i] = i;
        }
        int x = 0, tmp = 0;
        for (int i = length - 1; i > 0; i--) {
            //x = random.next(0, i + 1);
            //x = xorshf96() % i;
            x = uidis(engine) % i;
            tmp = array[i];
            array[i] = array[x];
            array[x] = tmp;
        }
        return array;
    }

    //模拟
    Node * simulate(Node* n) {
        ++(n->N);
        if (n->isTerminated) {
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

        int cnt = 0;

        int thisBoard[15][15];
        //用新棋盘模拟
        memcpy(thisBoard, board, sizeof(thisBoard));
        bool board_Flag[225];
        memset(board_Flag, false, sizeof(board_Flag));

        for (int i = 0; i < 15; ++i) {
            for (int j = 0; j < 15; ++j) {
                if (thisBoard[i][j]) {
                    ++cnt;
                    board_Flag[i * 15 + j] = true;
                }
            }
        }
        int pos = 0, i = 0, x, y;
        //一次性模拟到终局
        int* rand_perm = buildRandomSequence(SIZE*SIZE);

        while (cnt <= 225) {
            for (; i<225;) {
                if (board_Flag[rand_perm[i]]){
                    //已经存在则取下一个
                    i++;
                }
                else{
                    pos = rand_perm[i];
                    x = pos / 15;
                    y = pos % 15;
                    i++;
                    break;
                }
            }
            int jg = judge(thisBoard, own, x, y);
            ++cnt;
            thisBoard[x][y] = own;
            if (jg) {
                n->success = jg;
                if (jg == 1) ++(n->QB);
                else if (jg == 2) ++(n->QW);
                thisBoard[x][y] = 3;//这步没必要
                return n;
            }
            //交换下次下棋的颜色
            own ^= opp;
            opp ^= own;
            own ^= opp;
            pos = 0;
        }
        n->success = -1;
        return n;
    }


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
    void printNode(Node * r) {
        if (!r) return;
        cout << "color " << r->color << " x " << r->x << " y " << r->y << " QB/QW/N " << r->QB << " " << r->QW << " " << r->N << " isTerminated " << r->isTerminated << " depth " << r->depth << "score:" << (double)(r->score) / (double)(r->levelScore) << endl;
        for (Node *u = r->first; u; u = u->nxt) {
            //printNode(u);
            cout << "color " << u->color << " x " << u->x << " y " << u->y << " QB/QW/N " << u->QB << " " << u->QW << " " << u->N << " isTerminated " << u->isTerminated << " depth " << u->depth << "score:" << (double)(u->score) / (double)(u->levelScore) << endl;
        }
    }

    //玩家1--黑子
    int Black_Player(){
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
    int White_Player(){
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
    //模拟对弈
    //LocalSimulate
    void LocalSimulate(){
        //FILE *stream1;
        //freopen_s(&stream1, "out.txt", "w", stdout);
        //加载模型


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
    }
}