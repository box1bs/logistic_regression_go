package lr

import (
	"io"
	"os"
	"sort"
	"strconv"
	"strings"
)

type robustScaler struct {
	Q2, irq vec64
}

func RobustScaler() *robustScaler {
	return &robustScaler{}
}

func median(X_sorted vec64, n int) float64 {
	mid := n / 2
	if n%2 == 0 {
		return (X_sorted[mid-1] + X_sorted[mid]) / 2
	}
	return X_sorted[mid]
}

// p = (0.0, 1.0)
func percentile(X_sorted vec64, p float64, n int) float64 {
	if n == 1 {
		return X_sorted[0]
	}

	rank := p * float64(n-1)
	low := int(rank)
	high := low + 1
	if high >= n {
		return X_sorted[low]
	}

	weight := rank - float64(low)
	return X_sorted[low]*(1-weight) + X_sorted[high]*weight
}

func (rs *robustScaler) Fit(X []vec64) {
	samplesNum, featuresNum := len(X), len(X[0])
	transposed := Transpose(X)

	rs.Q2, rs.irq = make(vec64, featuresNum), make(vec64, featuresNum)
	for i := range featuresNum {
		sort.Float64s(transposed[i])
		if irq := percentile(transposed[i], 0.75, samplesNum) - percentile(transposed[i], 0.25, samplesNum); irq == 0 {
			rs.irq[i] = 1
		} else {
			rs.irq[i] = irq
		}
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
	file, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY, 0600)
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
		if i+1 < l {
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