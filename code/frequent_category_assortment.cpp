#include <bits/stdc++.h>
using namespace std;
const double eps = 1e-15;
const double eps2 = 1e-8;
const double INF = 1e12;

inline int dcmp(long double a){
	if(fabs(a)<eps) return 0;
	if(a>0) return 1;
	return -1;
}

inline int dcmp2(double a){
	if(fabs(a)<eps2) return 0;
	if(a>0) return 1;
	return -1;
}

inline int Rand(){
	long long t;
	t = rand()*10000LL+rand()%10000;
	return t%100000000;
}

struct Edge{
	int from, to;
	long double cap,flow;
};
class Maxflow{
	public:
	vector<Edge> edges;
	vector<vector<int> > G;
	vector<int> cur,d;
	int s,t,n;
	// number of nodes is from 0 to n-1
	
	inline void AddEdge(int from, int to, long double cap){
		edges.push_back((Edge{from,to,cap,0}));
		edges.push_back((Edge{to,from,0,0}));
		int m = edges.size();
		G[from].push_back(m-2);
		G[to].push_back(m-1);
	}
	
	void init(int n){
		this->n = n;
		G.clear();
		G.resize(n);
		edges.clear();
		cur.resize(n);
		d.resize(n);
		for(int i=0; i<n; i++) d[i] = cur[i] = 0;
	}
	

	vector<int> p;
	long double Augment(){
		int x = t;
		long double a = INF;
		while(x!=s){
			Edge&e = edges[p[x]];
			a = min(a,e.cap-e.flow);
			x = edges[p[x]].from;
		}
		x = t;
		while(x!=s){
			edges[p[x]].flow += a;
			edges[p[x]^1].flow -= a;
			x = edges[p[x]].from;
		}
		return a;
	}
	long double maxflow(int s,int t){
		this->s = s, this->t = t;
		long double flow = 0;
		vector<int> num;
		num.resize(n+1);
		for(int i=0; i<n; i++) d[i] = cur[i] = 0;
		num[0] = n;
		p.resize(n);
		for(int i=0; i<edges.size(); i++) edges[i].flow = 0;
		int x = s;
		while(d[s]<n){
			if(x==t){
				flow += Augment();
				x = s;
			}
			int ok = 0;
			for(int i=cur[x]; i<G[x].size(); i++){
				Edge&e = edges[G[x][i]];
				if(dcmp(e.cap-e.flow)>0 && d[x] == d[e.to]+1){
					ok = 1;
					p[e.to] = G[x][i];
					cur[x] = i;
					x = e.to;
					break;
				}
			}
			if(!ok){
				int m = n-1;
				for(int i=0; i<G[x].size(); i++){
					Edge& e = edges[G[x][i]];
					if(dcmp(e.cap-e.flow)>0) m = min(m,d[e.to]);
				}
				if(--num[d[x]]==0) break;
				num[d[x] = m+1] ++;
				cur[x] = 0;
				if(x!=s) x=edges[p[x]].from;
			}
		}
		return flow;
	}
	long double flow(){
		long double ret = 0;
		for(int i=0; i<G[t].size(); i++) ret -= edges[G[t][i]].flow;
		return ret;
	}
	
	vector<int> min_cut(){
		vector<int> que;
        vector<bool> in;
        in.resize(n);
		que.push_back(s);
		for(int i=0; i<n; i++) in[i] = false;
		in[s] = true;
		int now=0;
		while(now<que.size()){
			int tmp = que[now++];
			for(int i=0; i<G[tmp].size(); i++) {
				if(dcmp(edges[G[tmp][i]].flow-edges[G[tmp][i]].cap)<0 && !in[edges[G[tmp][i]].to]) {
					in[edges[G[tmp][i]].to] = true;
					que.push_back(edges[G[tmp][i]].to);
				}
			}
		}
		return que;
	}
}Graph,GraphInverse;

string getTime()
{
    time_t timep;
    time (&timep);
	char tmp[64];
	strftime(tmp, sizeof(tmp), "%Y_%m_%d_%H_%M_%S",localtime(&timep) );
	return tmp;
}

const int predict_days = 1;
const int period = 3;
const int Cardinality = 3;
//const int num_try = 10;
const int assortment_size = 50;
const double weight[] ={1, 1, 1, 1, 1, 1};
const double threshold = 0.001;

void work(const string date, const int n, vector<vector<int>> skuOfOrder, vector<vector<int>> true_order, ofstream & output, double threshold, const string type){
		vector<vector<int>> C;
		for(int i=0; i<skuOfOrder.size(); i++){
			sort(skuOfOrder[i].begin(), skuOfOrder[i].end());
		}
		sort(skuOfOrder.begin(),skuOfOrder.end());
		for(int i=0; i<true_order.size(); i++){
			sort(true_order[i].begin(), true_order[i].end());
		}
		sort(true_order.begin(), true_order.end());
		C.clear();
		for(int i=0,j=0; i<skuOfOrder.size(); i=j){
			while(j<skuOfOrder.size() && skuOfOrder[j]==skuOfOrder[i])
				j++;
			if(j-i>=threshold*skuOfOrder.size())
				while(i<j){
					C.push_back(skuOfOrder[i]);
					i++;
				}
		}
		skuOfOrder = C;
		C.clear();
		for(int i=0,j=0; i<true_order.size(); i=j){
			while(j<true_order.size() && true_order[j]==true_order[i])
				j++;
			if(j-i>=threshold*true_order.size())
				while(i<j){
					C.push_back(true_order[i]);
					i++;
				}
		}
		true_order = C;
		
		
		double value = 0;
//		for (int tt = 0; tt < num_try; tt++){
		vector<vector<int>> predict_order = skuOfOrder;
//			for(int i = 0; i<c_pre; i++){
//				predict_order.push_back(skuOfOrder[Rand()%skuOfOrder.size()]);
//			}
		
		int k = assortment_size;
//			int M = c_pre;
		int M = predict_order.size();
		Graph.init(n+M+2);
		GraphInverse.init(n+M+2);
		double total = 0;
		for(int i = 0; i<M; i++){
			total += weight[predict_order[i].size()-1];
			Graph.AddEdge(n+i+1,n+M+1,weight[predict_order[i].size()-1]);
			GraphInverse.AddEdge(n+M+1,n+i+1,weight[predict_order[i].size()-1]);
			for(int j=0; j<predict_order[i].size(); j++) {
				int c = predict_order[i][j];
				Graph.AddEdge(c,n+i+1,INF);
				GraphInverse.AddEdge(n+i+1,c,INF);
			}
			
		}
		
		// .......................
		cout<<"total: "<<total<<endl;
		vector<int> v;
		double l = 0, r = total + 1;
		for(int i=1; i<=n; i++) {
			Graph.AddEdge(0,i,l);
			GraphInverse.AddEdge(i,0,r);
			v.push_back(Graph.edges.size()-2);
		}
		
		// y = a0+a1*l, y = b0+b1*r, y = c0+c1*mid
		double a0 = 0, a1 = total, b0 = total, b1 = 0, c0, c1;
		cout<<"have built graph"<<endl;
		while(1){
			if(dcmp2(b1-a1)==0) break;
			double mid = (a0-b0)/(b1-a1);
			if(dcmp2(l-mid)==0 || dcmp2(r-mid)==0) break;
//			cout<<l<<' '<<r<<' '<<mid<<endl;
			printf("%.12lf %.12lf %.12lf\n",l,r,mid);
			printf("info: %.12lf %.12lf %.12lf %.12lf\n",a0,a1,b0,b1);
			Maxflow G1 = Graph, G2 = GraphInverse;
			for(int i=0; i<v.size(); i++) {
				Graph.edges[v[i]].cap = mid;
				GraphInverse.edges[v[i]].cap = mid;
			}
			Graph.maxflow(0,n+M+1);
			vector<int> cut = Graph.min_cut();
			int count = 0;
			for(int i=0; i<cut.size(); i++) if(cut[i]>0 && cut[i]<=n) count++;
			if(count<=n-k) {
				GraphInverse = G2;
				l = mid;
				double flow = Graph.flow();
				a1 = n-count;
				a0 = flow - a1*mid;
			} else{
				r = mid;
				Graph = G1;
				GraphInverse.maxflow(n+M+1,0);
				double flow = GraphInverse.flow();
				cut = GraphInverse.min_cut();
				count = 0;
				for(int i=0; i<cut.size(); i++) if(cut[i]>0 && cut[i]<=n) count++;
				b1 = count;
				b0 = flow - b1*mid;
			}
		}
		
		vector<int> in;
		in.resize(1+n);
		for(int i=1; i<=n; i++) in[i] = 1;
		Graph.maxflow(0,n+M+1);
		vector<int> cut = Graph.min_cut();
		printf("lambada: %lf\n",l);
	//	printf("cut is:\n");
	//	for(int i=0; i<cut.size(); i++) printf("%d ",cut[i]); printf("\n");
		int count = 0;
		for(int i=0; i<cut.size(); i++) if(cut[i]>0 && cut[i]<=n){
			in[cut[i]] = 0;
			count ++;
		}
		
		GraphInverse.maxflow(n+M+1,0);
		cut = GraphInverse.min_cut();
		for(int i=0; i<cut.size(); i++) if(cut[i]>0 && cut[i]<=n) in[cut[i]]=2;
		for(int i=1; i<=n && count<n-k; i++) if(in[i]==1){
			in[i] = 0;
			count++;
		}
		// output selection of assortment
	//	printf("selection is: \n");
	//	for(int i=1; i<=n; i++) if(in[i]) printf("%d ",i); printf("\n");
		
		double loss = 0;
		for(int i=n+1; i<=n+M; i++){
			double c;
			bool sat = true;
			for(int j=0; j<Graph.G[i].size(); j++){
				int to = Graph.edges[Graph.G[i][j]].to;
				if(to==n+M+1)
					c = Graph.edges[Graph.G[i][j]].cap;
				else if(to<=n && !in[to])
					sat = false;	
			}
			if(!sat) loss += c;
		}
		printf("satisfication rate: %lf\n",1.-loss/total);
		double train_rate = 1. - loss/total;
//			output<<d[T]<<","<<1.-loss/total<<",";
//			for(int p=0; p<P; p++) input[p].close();
		
	
		total = loss = 0;
		int tmp = 0;
		for(int i=1; i<=n; i++) if(in[i]!=0) tmp++;
		assert(tmp<=k);
		for(int i=0; i<true_order.size(); i++){
			total += weight[true_order[i].size()-1];
			bool flag = true;
			for(int j=0; j<true_order[i].size(); j++) {
				if(!in[true_order[i][j]]) flag = false;
			}
			if(!flag) loss += weight[true_order[i].size()-1];
		}
		value = 1. - loss/total;
		printf("%lf\n",total-loss);
//		value += total-loss;
//		}
//		value /= num_try;
//		output<<d[T]<<","<<value<<endl;
		output<<date<<", "<<k<<", "<<threshold<<", "<<true_order.size()<<", "<<predict_order.size()<<", "<<type<<", "<<train_rate<<", "<<value<<endl;
}

int main(){
	
	string d[11] = {"201808", "201809", "201810", "201811", "201812", "201901",
                 "201902", "201903", "201904", "201905", "201906"};
		
	srand(time(0));
	
	ofstream output;
//	output.open("../result/category_assortment/subsample_weighted_"+to_string(assortment_size)+"_assortment.csv");
	output.open("../result/category_assortment/frequent_category_assortment_"+to_string(assortment_size)+"_"+to_string(threshold)+"_.csv");
	output<<"date, capacity, threshold, # of true frequent orders, # of predict frequent orders, type, rain fulfillment rate, test fulfillment rate"<<endl;
	for(int T=0; T<11; T++){
		
		
//		int M = 0, m[P], n, k;
		ifstream input;
		input.open("../data3/"+d[T]+".txt");
		assert(input.is_open());
		
		int n,m;
		input>>n>>m;
		vector<vector<int>> skuOfOrder;
		vector<vector<int>> true_order;
//		int c_pre = 0;
		for(int i = 0; i<m; i++){
//				cout<<"ggg"<<endl;
			int date, frequency, cardinality;
			input>>date>>frequency>>cardinality;
			vector<int> v;
			for(int j = 0; j<cardinality; j++){
				int tmp;
				input>>tmp;
				v.push_back(tmp);
			}
			if(cardinality > Cardinality) continue;
			if (date%100>7 && date%100<=7+predict_days) {
				true_order.push_back(v);
//				c_pre += frequency;
			}
			if(date%100<=7-period || date %100 > 7) continue;
			skuOfOrder.push_back(v);
		}
		input.close();
		cout<<"input done"<<endl;
		work(d[T], n, skuOfOrder, true_order, output, threshold, "history");
		for(int number_try=0; number_try<5; number_try++){
			input.open("../result/adaptive_v3_"+to_string(Cardinality)+
			to_string(period)+to_string(predict_days)+"/20/order_"+d[T]+"_"+to_string(number_try)+".txt");
//			cout<<"../result/adaptive_v3_"+to_string(Cardinality)+
//			to_string(period)+to_string(predict_days)+"/order_"+d[T]+"_"+to_string(number_try)+".txt"<<endl;
			assert(input.is_open());
			input>>n>>m;
			skuOfOrder.clear();
			for(int i = 0; i<m; i++){
	//				cout<<"ggg"<<endl;
				int date, frequency, cardinality;
				input>>frequency>>cardinality;
				vector<int> v;
				for(int j = 0; j<cardinality; j++){
					int tmp;
					input>>tmp;
					v.push_back(tmp);
				}
				skuOfOrder.push_back(v);
			}
			input.close();
			work(d[T], n, skuOfOrder, true_order, output, threshold, "rw+rank");
		}
			
	}
	
	output.close();

	return 0;
}
