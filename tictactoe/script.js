let board = ["","","","","","","","",""];
let human = "X";
let ai = "O";
let gameOver = false;

const cells = document.querySelectorAll(".cell");
const statusText = document.getElementById("status");

cells.forEach(cell=>{
cell.addEventListener("click",humanMove);
});

function humanMove(e){

let index=e.target.dataset.index;

if(board[index]!=="" || gameOver) return;

board[index]=human;
e.target.textContent=human;

if(checkWinner(board,human)){
statusText.textContent="You Win!";
gameOver=true;
return;
}

if(board.every(c=>c!="")){
statusText.textContent="Draw!";
return;
}

statusText.textContent="AI thinking...";

setTimeout(aiMove,500);

}

function aiMove(){

let bestScore=-Infinity;
let move;

for(let i=0;i<9;i++){

if(board[i]==""){

board[i]=ai;
let score=minimax(board,false);
board[i]="";

if(score>bestScore){

bestScore=score;
move=i;

}

}

}

board[move]=ai;
cells[move].textContent=ai;

if(checkWinner(board,ai)){
statusText.textContent="AI Wins!";
gameOver=true;
return;
}

if(board.every(c=>c!="")){
statusText.textContent="Draw!";
return;
}

statusText.textContent="Your turn (X)";

}

function minimax(board,isMaximizing){

if(checkWinner(board,ai)) return 1;
if(checkWinner(board,human)) return -1;
if(board.every(c=>c!="")) return 0;

if(isMaximizing){

let bestScore=-Infinity;

for(let i=0;i<9;i++){

if(board[i]==""){

board[i]=ai;
let score=minimax(board,false);
board[i]="";

bestScore=Math.max(score,bestScore);

}

}

return bestScore;

}

else{

let bestScore=Infinity;

for(let i=0;i<9;i++){

if(board[i]==""){

board[i]=human;
let score=minimax(board,true);
board[i]="";

bestScore=Math.min(score,bestScore);

}

}

return bestScore;

}

}

function checkWinner(board,player){

const patterns=[
[0,1,2],
[3,4,5],
[6,7,8],
[0,3,6],
[1,4,7],
[2,5,8],
[0,4,8],
[2,4,6]
];

return patterns.some(pattern =>
pattern.every(i=>board[i]==player)
);

}

function restartGame(){

board=["","","","","","","","",""];
gameOver=false;

cells.forEach(cell=>{
cell.textContent="";
});

statusText.textContent="Your turn (X)";

}
