package lr

import (
	"log"
	"math"
	"strconv"
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