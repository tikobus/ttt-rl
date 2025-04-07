package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

const (
	NN_INPUT_SIZE  = 18
	NN_HIDDEN_SIZE = 100
	NN_OUTPUT_SIZE = 9
	LEARNING_RATE  = 0.1
)

// GameState 表示游戏状态
type GameState struct {
	board         [9]rune
	currentPlayer int
}

// NeuralNetwork 表示神经网络
type NeuralNetwork struct {
	weightsIH [NN_INPUT_SIZE * NN_HIDDEN_SIZE]float64
	weightsHO [NN_HIDDEN_SIZE * NN_OUTPUT_SIZE]float64
	biasesH   [NN_HIDDEN_SIZE]float64
	biasesO   [NN_OUTPUT_SIZE]float64
	inputs    [NN_INPUT_SIZE]float64
	hidden    [NN_HIDDEN_SIZE]float64
	rawLogits [NN_OUTPUT_SIZE]float64
	outputs   [NN_OUTPUT_SIZE]float64
}

// relu 是 ReLU 激活函数
func relu(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

// reluDerivative 是 ReLU 激活函数的导数
func reluDerivative(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}

// initNeuralNetwork 初始化神经网络
func initNeuralNetwork(nn *NeuralNetwork) {
	for i := range nn.weightsIH {
		nn.weightsIH[i] = rand.Float64() - 0.5
	}
	for i := range nn.weightsHO {
		nn.weightsHO[i] = rand.Float64() - 0.5
	}
	for i := range nn.biasesH {
		nn.biasesH[i] = rand.Float64() - 0.5
	}
	for i := range nn.biasesO {
		nn.biasesO[i] = rand.Float64() - 0.5
	}
}

// softmax 是 Softmax 激活函数
func softmax(input []float64, output []float64, size int) {
	maxVal := input[0]
	for i := 1; i < size; i++ {
		if input[i] > maxVal {
			maxVal = input[i]
		}
	}
	sum := 0.0
	for i := 0; i < size; i++ {
		output[i] = math.Exp(input[i] - maxVal)
		sum += output[i]
	}
	if sum > 0 {
		for i := 0; i < size; i++ {
			output[i] /= sum
		}
	} else {
		for i := 0; i < size; i++ {
			output[i] = 1.0 / float64(size)
		}
	}
}

// forwardPass 进行神经网络前向传播
func forwardPass(nn *NeuralNetwork, inputs []float64) {
	copy(nn.inputs[:], inputs)
	for i := 0; i < NN_HIDDEN_SIZE; i++ {
		sum := nn.biasesH[i]
		for j := 0; j < NN_INPUT_SIZE; j++ {
			sum += inputs[j] * nn.weightsIH[j*NN_HIDDEN_SIZE+i]
		}
		nn.hidden[i] = relu(sum)
	}
	for i := 0; i < NN_OUTPUT_SIZE; i++ {
		nn.rawLogits[i] = nn.biasesO[i]
		for j := 0; j < NN_HIDDEN_SIZE; j++ {
			nn.rawLogits[i] += nn.hidden[j] * nn.weightsHO[j*NN_OUTPUT_SIZE+i]
		}
	}
	softmax(nn.rawLogits[:], nn.outputs[:], NN_OUTPUT_SIZE)
}

// initGame 初始化游戏状态
func initGame(state *GameState) {
	for i := range state.board {
		state.board[i] = '.'
	}
	state.currentPlayer = 0
}

// displayBoard 显示棋盘
func displayBoard(state *GameState) {
	for row := 0; row < 3; row++ {
		fmt.Printf("%c%c%c ", state.board[row*3], state.board[row*3+1], state.board[row*3+2])
		fmt.Printf("%d%d%d\n", row*3, row*3+1, row*3+2)
	}
	fmt.Println()
}

// boardToInputs 将棋盘状态转换为神经网络输入
func boardToInputs(state *GameState, inputs []float64) {
	for i := 0; i < 9; i++ {
		if state.board[i] == '.' {
			inputs[i*2] = 0
			inputs[i*2+1] = 0
		} else if state.board[i] == 'X' {
			inputs[i*2] = 1
			inputs[i*2+1] = 0
		} else {
			inputs[i*2] = 0
			inputs[i*2+1] = 1
		}
	}
}

// checkGameOver 检查游戏是否结束
func checkGameOver(state *GameState, winner *rune) bool {
	// 检查行
	for i := 0; i < 3; i++ {
		if state.board[i*3] != '.' && state.board[i*3] == state.board[i*3+1] && state.board[i*3+1] == state.board[i*3+2] {
			*winner = state.board[i*3]
			return true
		}
	}
	// 检查列
	for i := 0; i < 3; i++ {
		if state.board[i] != '.' && state.board[i] == state.board[i+3] && state.board[i+3] == state.board[i+6] {
			*winner = state.board[i]
			return true
		}
	}
	// 检查对角线
	if state.board[0] != '.' && state.board[0] == state.board[4] && state.board[4] == state.board[8] {
		*winner = state.board[0]
		return true
	}
	if state.board[2] != '.' && state.board[2] == state.board[4] && state.board[4] == state.board[6] {
		*winner = state.board[2]
		return true
	}
	// 检查平局
	emptyTiles := 0
	for i := range state.board {
		if state.board[i] == '.' {
			emptyTiles++
		}
	}
	if emptyTiles == 0 {
		*winner = 'T'
		return true
	}
	return false
}

// getComputerMove 获取计算机的最佳落子位置
func getComputerMove(state *GameState, nn *NeuralNetwork, displayProbs bool) int {
	inputs := make([]float64, NN_INPUT_SIZE)
	boardToInputs(state, inputs)
	forwardPass(nn, inputs)

	highestProb := -1.0
	highestProbIdx := -1
	bestMove := -1
	bestLegalProb := -1.0

	for i := 0; i < 9; i++ {
		if nn.outputs[i] > highestProb {
			highestProb = nn.outputs[i]
			highestProbIdx = i
		}
		if state.board[i] == '.' && (bestMove == -1 || nn.outputs[i] > bestLegalProb) {
			bestMove = i
			bestLegalProb = nn.outputs[i]
		}
	}

	if displayProbs {
		fmt.Println("Neural network move probabilities:")
		for row := 0; row < 3; row++ {
			for col := 0; col < 3; col++ {
				pos := row*3 + col
				fmt.Printf("%5.1f%%", nn.outputs[pos]*100.0)
				if pos == highestProbIdx {
					fmt.Print("*")
				}
				if pos == bestMove {
					fmt.Print("#")
				}
				fmt.Print(" ")
			}
			fmt.Println()
		}
		totalProb := 0.0
		for i := 0; i < 9; i++ {
			totalProb += nn.outputs[i]
		}
		fmt.Printf("Sum of all probabilities: %.2f\n\n", totalProb)
	}
	return bestMove
}

// backprop 进行反向传播更新神经网络权重
func backprop(nn *NeuralNetwork, targetProbs []float64, learningRate float64, rewardScaling float64) {
	outputDeltas := make([]float64, NN_OUTPUT_SIZE)
	hiddenDeltas := make([]float64, NN_HIDDEN_SIZE)

	// 计算输出层误差
	for i := 0; i < NN_OUTPUT_SIZE; i++ {
		outputDeltas[i] = (nn.outputs[i] - targetProbs[i]) * math.Abs(rewardScaling)
	}
	// 反向传播误差到隐藏层
	for i := 0; i < NN_HIDDEN_SIZE; i++ {
		error := 0.0
		for j := 0; j < NN_OUTPUT_SIZE; j++ {
			error += outputDeltas[j] * nn.weightsHO[i*NN_OUTPUT_SIZE+j]
		}
		hiddenDeltas[i] = error * reluDerivative(nn.hidden[i])
	}
	// 更新输出层权重和偏置
	for i := 0; i < NN_HIDDEN_SIZE; i++ {
		for j := 0; j < NN_OUTPUT_SIZE; j++ {
			nn.weightsHO[i*NN_OUTPUT_SIZE+j] -= learningRate * outputDeltas[j] * nn.hidden[i]
		}
	}
	for j := 0; j < NN_OUTPUT_SIZE; j++ {
		nn.biasesO[j] -= learningRate * outputDeltas[j]
	}
	// 更新隐藏层权重和偏置
	for i := 0; i < NN_INPUT_SIZE; i++ {
		for j := 0; j < NN_HIDDEN_SIZE; j++ {
			nn.weightsIH[i*NN_HIDDEN_SIZE+j] -= learningRate * hiddenDeltas[j] * nn.inputs[i]
		}
	}
	for j := 0; j < NN_HIDDEN_SIZE; j++ {
		nn.biasesH[j] -= learningRate * hiddenDeltas[j]
	}
}

// learnFromGame 从游戏结果中学习
func learnFromGame(nn *NeuralNetwork, moveHistory []int, numMoves int, nnMovesEven bool, winner rune) {
	var reward float64
	nnSymbol := 'O'
	if !nnMovesEven {
		nnSymbol = 'X'
	}
	if winner == 'T' {
		reward = 0.3
	} else if winner == nnSymbol {
		reward = 1.0
	} else {
		reward = -2.0
	}

	state := GameState{}
	targetProbs := make([]float64, NN_OUTPUT_SIZE)

	for moveIdx := 0; moveIdx < numMoves; moveIdx++ {
		if (nnMovesEven && moveIdx%2 != 1) || (!nnMovesEven && moveIdx%2 != 0) {
			continue
		}
		initGame(&state)
		for i := 0; i < moveIdx; i++ {
			symbol := 'X'
			if i%2 == 1 {
				symbol = 'O'
			}
			state.board[moveHistory[i]] = symbol
		}
		inputs := make([]float64, NN_INPUT_SIZE)
		boardToInputs(&state, inputs)
		forwardPass(nn, inputs)

		move := moveHistory[moveIdx]
		moveImportance := 0.5 + 0.5*float64(moveIdx)/float64(numMoves)
		scaledReward := reward * moveImportance

		for i := range targetProbs {
			targetProbs[i] = 0
		}
		if scaledReward >= 0 {
			targetProbs[move] = 1
		} else {
			validMovesLeft := 9 - moveIdx - 1
			otherProb := 1.0 / float64(validMovesLeft)
			for i := 0; i < 9; i++ {
				if state.board[i] == '.' && i != move {
					targetProbs[i] = otherProb
				}
			}
		}
		backprop(nn, targetProbs, LEARNING_RATE, scaledReward)
	}
}

// getRandomMove 获取随机合法落子位置
func getRandomMove(state *GameState) int {
	for {
		move := rand.Intn(9)
		if state.board[move] == '.' {
			return move
		}
	}
}

// playRandomGame 与随机对手进行一场游戏并学习
func playRandomGame(nn *NeuralNetwork, moveHistory []int, numMoves *int) rune {
	state := GameState{}
	var winner rune
	*numMoves = 0
	initGame(&state)

	for !checkGameOver(&state, &winner) {
		var move int
		if state.currentPlayer == 0 {
			move = getRandomMove(&state)
		} else {
			move = getComputerMove(&state, nn, false)
		}
		symbol := 'X'
		if state.currentPlayer == 1 {
			symbol = 'O'
		}
		state.board[move] = symbol
		moveHistory[*numMoves] = move
		(*numMoves)++
		state.currentPlayer = 1 - state.currentPlayer
	}
	learnFromGame(nn, moveHistory, *numMoves, true, winner)
	return winner
}

// trainAgainstRandom 训练神经网络与随机对手进行多场游戏
func trainAgainstRandom(nn *NeuralNetwork, numGames int) {
	moveHistory := make([]int, 9)
	var numMoves int
	wins, losses, ties := 0, 0, 0

	fmt.Printf("Training neural network against %d random games...\n", numGames)
	for i := 0; i < numGames; i++ {
		winner := playRandomGame(nn, moveHistory, &numMoves)
		if winner == 'O' {
			wins++
		} else if winner == 'X' {
			losses++
		} else {
			ties++
		}
		if (i+1)%10000 == 0 {
			fmt.Printf("Games: %d, Wins: %d (%.1f%%), Losses: %d (%.1f%%), Ties: %d (%.1f%%)\n",
				i+1, wins, float64(wins)*100/float64(i+1),
				losses, float64(losses)*100/float64(i+1),
				ties, float64(ties)*100/float64(i+1))
		}
	}
	fmt.Println("\nTraining complete!")
}

// playGame 与人类玩家进行一场游戏
func playGame(nn *NeuralNetwork) {
	state := GameState{}
	var winner rune
	moveHistory := make([]int, 9)
	var numMoves int
	initGame(&state)

	fmt.Println("Welcome to Tic Tac Toe! You are X, the computer is O.")
	fmt.Println("Enter positions as numbers from 0 to 8 (see picture).")

	for !checkGameOver(&state, &winner) {
		displayBoard(&state)
		if state.currentPlayer == 0 {
			var move int
			fmt.Print("Your move (0-8): ")
			fmt.Scan(&move)
			if move < 0 || move > 8 || state.board[move] != '.' {
				fmt.Println("Invalid move! Try again.")
				continue
			}
			state.board[move] = 'X'
			moveHistory[numMoves] = move
			numMoves++
		} else {
			fmt.Println("Computer's move:")
			move := getComputerMove(&state, nn, true)
			state.board[move] = 'O'
			fmt.Printf("Computer placed O at position %d\n", move)
			moveHistory[numMoves] = move
			numMoves++
		}
		state.currentPlayer = 1 - state.currentPlayer
	}
	displayBoard(&state)
	if winner == 'X' {
		fmt.Println("You win!")
	} else if winner == 'O' {
		fmt.Println("Computer wins!")
	} else {
		fmt.Println("It's a tie!")
	}
	learnFromGame(nn, moveHistory, numMoves, true, winner)
}

func main() {
	rand.Seed(time.Now().UnixNano())
	nn := NeuralNetwork{}
	initNeuralNetwork(&nn)

	randomGames := 150000
	trainAgainstRandom(&nn, randomGames)
	playGame(&nn)
}
