package exec

import (
	"fmt"

	"github.com/pkg/errors"

	"github.com/nikolalohinski/gonja/v2/nodes"
)

// TestFunction is the type test functions must fulfill is
// type TestFunction func(*Evaluator, *Value, *VarArgs) (bool, error)
// but we use an so as to support the legacy type
// type TestFunction func(*Context, *Value, *VarArgs) (bool, error)
// in a backwards compatible way
type TestFunction any

func (e *Evaluator) EvalTest(expr *nodes.TestExpression) *Value {
	value := e.Eval(expr.Expression)

	return e.ExecuteTest(expr.Test, value)
}

func (e *Evaluator) ExecuteTest(tc *nodes.TestCall, v *Value) *Value {
	params := &VarArgs{
		Args:   []*Value{},
		KwArgs: map[string]*Value{},
	}

	for _, param := range tc.Args {
		value := e.Eval(param)
		if value.IsError() {
			return AsValue(errors.Wrapf(value, `Unable to evaluate parameter %s`, param))
		}
		params.Args = append(params.Args, value)
	}

	for key, param := range tc.Kwargs {
		value := e.Eval(param)
		if value.IsError() {
			return AsValue(errors.Wrapf(value, `Unable to evaluate parameter %s`, param))
		}
		params.KwArgs[key] = value
	}

	return e.ExecuteTestByName(tc.Name, v, params)
}

func (e *Evaluator) ExecuteTestByName(name string, in *Value, params *VarArgs) *Value {
	test, ok := e.Environment.Tests.Get(name)
	if !e.Environment.Tests.Exists(name) || !ok {
		return AsValue(errors.Errorf("test '%s' not found", name))
	}

	if err := e.Environment.Tests.validate(name, test); err != nil {
		return AsValue(fmt.Errorf("test '%s' is invalid: %q", name, err))
	}

	// Use typed dispatch instead of reflect.Call to avoid Go 1.26 reflection
	// ABI issues that cause "slice bounds out of range" panics.
	var result bool
	var err error

	switch fn := test.(type) {
	case func(*Evaluator, *Value, *VarArgs) (bool, error):
		result, err = fn(e, in, params)
	case func(*Context, *Value, *VarArgs) (bool, error):
		result, err = fn(e.Environment.Context, in, params)
	default:
		return AsValue(fmt.Errorf("test '%s' has unsupported signature %T", name, test))
	}

	if callErr, ok := err.(ErrInvalidCall); ok && err != nil {
		return AsValue(fmt.Errorf("invalid call to test '%s': %s", name, callErr.Error()))
	} else if err != nil {
		return AsValue(fmt.Errorf("unable to execute test '%s': %s", name, err.Error()))
	} else {
		return AsValue(result)
	}
}
