package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"reflect"
	"reflect"
	"sort"
	"strconv"
	"strings"
)

type vec64 []float64
type numeric interface {
	float64 | float32 | int | int8 | int16 | int32 | int64 | uint | uint8 | uint16 | uint32 | uint64
}

func Vec[T string | numeric](x ...T) vec64 {
	v := make(vec64, 0, len(x))
	for _, f := range x {
		switch tvalue := any(f).(type) {
		case string:
			parsed, err := strconv.ParseFloat(tvalue, 64)
			if err != nil {
				continue
			}
			v = append(v, parsed)
		case float64:
			v = append(v, tvalue)
		case float32:
			v = append(v, float64(tvalue))
		case int64:
			v = append(v, float64(tvalue))
		case int8:
			v = append(v, float64(tvalue))
		case int16:
			v = append(v, float64(tvalue))
		case int32:
			v = append(v, float64(tvalue))
		case int:
			v = append(v, float64(tvalue))
		case uint:
			v = append(v, float64(tvalue))
		case uint8:
			v = append(v, float64(tvalue))
		case uint16:
			v = append(v, float64(tvalue))
		case uint32:
			v = append(v, float64(tvalue))
		case uint64:
			v = append(v, float64(tvalue))
		default:
			log.Printf("unsupported type: %T\n", f)
		}
	}
	return v
}

func (v vec64) Add(scalar float64) vec64 {
	newv := make(vec64, len(v))
	for i := range v {
		newv[i] = v[i] + scalar
	}
	return newv
}

func (v vec64) Subtract(scalar float64) vec64 { return v.Add(-scalar) }

func (v vec64) Dot(scalar float64) vec64 {
	newv := make(vec64, len(v))
	for i := range v {
		newv[i] = v[i] * scalar
	}
	return newv
}

func (v vec64) Mean() float64 {
	return v.Sum() / float64(len(v))
}

func (v vec64) Divide(scalar float64) vec64 {
	newv := make(vec64, len(v))
	for i := range v {
		newv[i] = v[i] / scalar
	}
	return newv
}

func (v vec64) Sum() float64 {
	sum := 0.0
	for i := range v {
		sum += v[i]
	}
	return sum
}

func (v vec64) Pow(scale float64) vec64 {
	newv := make(vec64, len(v))
	for i := range v {
		newv[i] = math.Pow(v[i], scale)
	}
	return newv
}

func (v vec64) L2Normalization() vec64 {
	newv := make(vec64, len(v))
	l2Norm := math.Sqrt(v.Pow(2.0).Sum())
	for i := range v {
		newv[i] = v[i] / l2Norm
	}
	return newv
}

func Add(a, b vec64) vec64 {
	if len(a) != len(b) {
		log.Fatalf("vectors must be of the same length: len(a)=%d, len(b)=%d\n", len(a), len(b))
		return nil
	}
	result := make(vec64, len(a))
	for i := range a {
		result[i] = a[i] + b[i]
	}
	return result
}

func Subtract(a, b vec64) vec64 {
	if len(a) != len(b) {
		log.Fatalf("vectors must be of the same length: len(a)=%d, len(b)=%d\n", len(a), len(b))
		return nil
	}
	result := make(vec64, len(a))
	for i := range a {
		result[i] = a[i] - b[i]
	}
	return result
}

func Dot(a, b vec64) float64 {
	if len(a) != len(b) {
		log.Fatalf("vectors must be of the same length: len(a)=%d, len(b)=%d\n", len(a), len(b))
		return 0
	}
	sum := 0.0
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

func sigmoid(z float64) float64 {
	return 1 / (1 + math.Exp(-z))
}

func computeGradient(X []vec64, y vec64, w vec64, b float64) (vec64, float64) {
	samplesNum, featuresNum := len(y), len(X[0])
	dj_dw := make(vec64, featuresNum)
	dj_db := 0.0

	for i := range samplesNum {
		yhati := sigmoid(Dot(X[i], w) + b)
		err := yhati - y[i]
		for j := range featuresNum {
			dj_dw[j] += err * X[i][j]
		}
		dj_db += err
	}

	return dj_dw.Divide(float64(samplesNum)), dj_db / float64(samplesNum)
}

func computeGradientReg(X []vec64, y vec64, w vec64, b, lambda float64) (vec64, float64) {
	samplesNum, featuresNum := len(y), len(X[0])
	dj_dw, dj_db := computeGradient(X, y, w, b)

	if lambda != 0 {
		for j := range featuresNum {
			dj_dw[j] += lambda / float64(samplesNum) * w[j]
		}
	}

	return dj_dw, dj_db
}

type queue struct {
	head, best, end *node
	len, cap int
}

type node struct {
	next *node
	w vec64
	b float64
	val  float64
}

func (q *queue) insert(w vec64, b, val float64) {
	if q.len == 0 {q.head=&node{w: w, b: b, val: val};q.best=q.head; q.end=q.head; return}
	if q.len == q.cap {
		q.delete()
	}
	q.end.next = &node{w: w, b: b, val: val}
	q.end = q.end.next
	if q.best.val > q.end.val {
		q.best = q.end
	}
	q.len++
}

func (q *queue) delete() {
	if q.best == q.head {
		p := q.head.next
		q.best = p
		for p != nil {
			if p.val < q.best.val {
				q.best = p
			}
			p = p.next
		}
	}
	q.head = q.head.next
	q.len--
}

func gradientDescent(X []vec64, y, wInit vec64, bInit, learningRate float64, numIters, checkEach, earlyStopping int) (vec64, float64) {
	w := make(vec64, len(wInit))
	copy(w, wInit)
	b := bInit
	q := queue{cap: earlyStopping}
	for i := range numIters {
		dj_dw, dj_db := computeGradient(X, y, w, b)

		w = Subtract(w, dj_dw.Dot(learningRate))
		b -= learningRate * dj_db

		cost := BinaryCrossEntropy(X, y, w, b)
		q.insert(w, b, cost)
		if (i+1)%checkEach == 0 {
			log.Printf("Iteration: %d, cost: %.4f\n", i+1, cost)
		}
		if i > earlyStopping && q.head.val <= cost {
			log.Printf("early stopped after %d iterations, with best score: %.3f", i, q.best.val)
			return q.best.w, q.best.b
		}
	}

	return w, b
}

func gradientDescentReg(X []vec64, y, wInit vec64, bInit, learningRate, lambda float64, numIters, checkEach, earlyStopping int) (vec64, float64) {
	w := make(vec64, len(wInit))
	copy(w, wInit)
	b := bInit
	q := queue{cap: earlyStopping}
	for i := range numIters {
		dj_dw, dj_db := computeGradientReg(X, y, w, b, lambda)

		w = Subtract(w, dj_dw.Dot(learningRate))
		b -= learningRate * dj_db

		cost := BinaryCrossEntropy(X, y, w, b, lambda)
		q.insert(w, b, cost)
		if (i+1)%checkEach == 0 {
			log.Printf("Iteration: %d, cost: %.4f\n", i+1, cost)
		}
		if i > earlyStopping && q.head.val < cost {
			log.Printf("early stopped after %d iterations, with best score: %.3f", i, q.best.val)
			return q.best.w, q.best.b
		}
	}

	return w, b
}

func BinaryCrossEntropy(X []vec64, y, w vec64, b, lambda float64) float64 {
	cost := 0.0
	samples := len(y)
	epsilon := 1e-15
	for i := range samples {
		yhati := sigmoid(Dot(X[i], w) + b)
		cost += -y[i]*math.Log(yhati + epsilon) - (1-y[i])*math.Log(1-yhati+epsilon)
	}
	if lambda != 0 {
		cost += lambda/(float64(samples)*2)*w.Pow(2.0).Sum()
	}
	return cost / float64(samples)
}

func UploadDataset(path string, hasNamedCols bool) ([]vec64, vec64, error) {
	file, err := os.OpenFile(path, os.O_RDONLY, 0600)
	if err != nil {
		return nil, nil, err
	}
	defer file.Close()
	r := csv.NewReader(file)
	records, err := r.ReadAll()
	if err != nil {
		return nil, nil, err
	}
	if hasNamedCols {
		records = records[1:]
	}
	X := make([]vec64, 0, 64)
	scores := make([]string, 0, 64)
	shape := len(records[0])
	for i := range records {
		X = append(X, Vec(records[i][:shape - 1]...))
		scores = append(scores, records[i][shape - 1])
	}
	y := Vec(scores...)
	return X, y, nil
}

type robustScaler struct {
	Q2, irq vec64
}

func RobustScaler() *robustScaler {
	return &robustScaler{}
}

func median(X_sorted vec64, n int) float64 {
	mid := n / 2
	if n % 2 == 0 {
		return (X_sorted[mid - 1] + X_sorted[mid]) / 2
	}
	return X_sorted[mid]
}

// p = (0.0, 1.0)
func percentile(X_sorted vec64, p float64, n int) float64 {
	if n == 1 {return X_sorted[0]}

	rank := p * float64(n - 1)
	low := int(rank)
	high := low + 1
	if high >= n {return X_sorted[low]}

	weight := rank - float64(low)
	return X_sorted[low] * (1 - weight) + X_sorted[high] * weight
}

func (rs *robustScaler) Fit(X []vec64) {
	samplesNum, featuresNum := len(X), len(X[0])
	transposed := Transpose(X)

	rs.Q2, rs.irq = make(vec64, featuresNum), make(vec64, featuresNum)
	for i := range featuresNum {
		sort.Float64s(transposed[i])
		if irq := percentile(transposed[i], 0.75, samplesNum) - percentile(transposed[i], 0.25, samplesNum); irq == 0 {
			rs.irq[i] = 1
		} else { rs.irq[i] = irq }
		rs.Q2[i] = median(transposed[i], samplesNum)
	}
}

func (rs *robustScaler) Scale2D(matrix []vec64) {
	samplesNum := len(matrix)
	for i := range samplesNum {
		rs.Scale1D(matrix[i])
	}
}

func (rs *robustScaler) Scale1D(vec vec64) {
	featuresNum := len(vec)
	for i := range featuresNum {
		vec[i] = (vec[i] - rs.Q2[i]) / rs.irq[i]
	}
}

func (rs *robustScaler) LoadToFile(path string) error {
	file, err := os.OpenFile(path, os.O_CREATE | os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	defer file.Close()
	sb := strings.Builder{}
	l := len(rs.irq)
	for i := range l {
		sb.WriteString(strconv.FormatFloat(rs.Q2[i], 'f', 16, 64))
		sb.WriteByte(':')
		sb.WriteString(strconv.FormatFloat(rs.irq[i], 'f', 16, 64))
		if i + 1 < l {
			sb.WriteByte(',')
		}
	}
	_, err = file.WriteString(sb.String())
	return err
}

func (rs *robustScaler) LoadFromFile(path string) (*robustScaler, error) {
	file, err := os.OpenFile(path, os.O_RDONLY, 0600)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	medians, irqs := []string{}, []string{}
	body, err := io.ReadAll(file)
	if err != nil {
		return nil, err
	}
	sets := strings.SplitSeq(string(body), ",")
	for set := range sets {
		pair := strings.Split(set, ":")
		medians = append(medians, pair[0])
		irqs = append(irqs, pair[1])
	}
	rs.Q2 = Vec(medians...)
	rs.irq = Vec(irqs...)
	return rs, nil
}

func Transpose(v []vec64) []vec64 {
	newm := make([]vec64, 0, 8)
	rows, cols := len(v), len(v[0])
	for i := range cols {
		newv := make(vec64, rows)
		for j := range rows {
			newv[j] = v[j][i]
		}
		newm = append(newm, newv)
	}
	return newm
}

type model struct {
	w vec64
	b float64
	threshold float64
}

// threshold = (0.1, 1.0)
func LogisticRegressor(threshold float64) *model {
	return &model{threshold: threshold}
}

func (m *model) LoadToFile(path string) error {
	file, err := os.OpenFile(path, os.O_CREATE | os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	defer file.Close()
	sb := strings.Builder{}
	for i := range m.w {
		sb.WriteString(strconv.FormatFloat(m.w[i], 'f', 32, 64))
		sb.WriteByte(',')
	}
	sb.WriteString(strconv.FormatFloat(m.b, 'f', 32, 64))
	_, err = file.WriteString(sb.String())
	return err
}

func (m *model) LoadFromFile(path string) error {
	file, err := os.OpenFile(path, os.O_RDONLY, 0600)
	if err != nil {
		return err
	}
	defer file.Close()
	body, err := io.ReadAll(file)
	if err != nil {
		return err
	}
	vec := Vec(strings.Split(string(body), ",")...)
	last := len(vec) - 1
	m.w = vec[:last]
	m.b = vec[last]
	return nil
}

// lambda_ = 0 for disable reg
func (m *model) Fit(X []vec64, y vec64, lambda_ float64, randomState, epochs, printAfterEach, earlyStopping int) {
	featuren := len(X[0])
	wInit := make(vec64, featuren)
	rd := rand.New(rand.NewSource(int64(randomState)))
	for i := range featuren {
		wInit[i] = rd.Float64()
	}
	bInit := rd.Float64()
	lrate := 0.05

	m.w, m.b = gradientDescent(X, y, wInit, bInit, lrate, lambda_, epochs, printAfterEach, earlyStopping)
}

func (m *model) Predict(X ...vec64) vec64 {
	rows := len(X)
	p := make(vec64, rows)
	for i := range rows {
		if sigmoid(Dot(X[i], m.w) + m.b) > m.threshold {p[i] = 1; continue}
		p[i] = 0
	}
	return p
}

func ComputePredictMetrics(yhat, y vec64) (tp, tn, fp, fn float64) {
	for i := range y {
		if y[i] == 1 {
			if y[i] == yhat[i] {
				tp++
			} else {
				fn++
			}
		} else {
			if y[i] == yhat[i] {
				tn++
			} else {
				fp++
			}
		}
	}
	return
}

func SplitSample(X []vec64, y vec64, scale float64) ([]vec64, []vec64, vec64, vec64) {
	sampleNum := len(X)
	scaled := int(float64(sampleNum) * scale)
	r := rand.New(rand.NewSource(42))
	generated := r.Perm(sampleNum)
	trainIdx := generated[:scaled]
	testIdx := generated[scaled:]
	trainX, testX, trainY, testY := make([]vec64, scaled), make([]vec64, sampleNum - scaled), make(vec64, scaled), make(vec64, sampleNum - scaled)
	for i := range scaled {
		trainX[i], trainY[i] = X[trainIdx[i]], y[trainIdx[i]]
		if i < sampleNum - scaled {
			testX[i], testY[i] = X[testIdx[i]], y[testIdx[i]]
		}
	}
	return trainX, testX, trainY,testY
}

func main() {
	X, y, err := UploadDataset("html.csv", true)
	if err != nil {
		panic(err)
	}
	scaler := robustScaler{}
	scaler.Fit(X)
	scaler.Scale2D(X)
	Xtrain, Xtest, ytrain, ytest := SplitSample(X, y, 0.8)
	lr := LogisticRegressor(0.4)
	lr.Fit(Xtrain, ytrain, 1, 42, 200000, 100, 100)
<<<<<<< HEAD
	yhatT := lr.Predict(Xtest...)
	tp, tn, fp, fn := ComputePredictMetrics(yhatT, ytest)
	prec, rec := tp / (tp + fp), tp / (tp + fn)
	fmt.Printf("Precision: %.4f\n", prec)
	fmt.Printf("Recall: %.4f\n", rec)
	fmt.Printf("Accuracy: %.4f\n", (tp + tn) / (tp + tn + fp + fn))
	fmt.Printf("F1-score: %.4f\n", 2 * prec * rec / (prec + rec))
=======
>>>>>>> 6d9d2d8 (генерализировал регуляризацию, пофиксил права загрузки в файл)
	fmt.Println(lr)
	fmt.Printf("log loss: %.4f\n", BinaryCrossEntropy(Xtest, ytest, lr.w, lr.b, 0))
	fmt.Printf("reg log loss: %.4f\n", BinaryCrossEntropy(Xtest, ytest, lr.w, lr.b, 1))
	if err := lr.LoadToFile("lr1"); err != nil {panic(err)}
	tmp_w, tmp_b := lr.w, lr.b
	if err := lr.LoadFromFile("lr1"); err != nil {panic(err)}
	if !reflect.DeepEqual(tmp_w, lr.w) || !reflect.DeepEqual(tmp_b, lr.b) {panic("unexpected behavior")}
}