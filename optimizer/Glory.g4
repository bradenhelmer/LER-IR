grammar Glory;

statement: l e '=' r
;

l: loopType = (NORMAL|SUMMATION|PRODUCT) forParam l
| loopType = OTHER subscript conditionExpression '∫' l
|  // this should be expilon
;

NORMAL: 'Γ'
;

SUMMATION: 'Σ'
;

PRODUCT: 'Π'
;

OTHER: 'Ψ'
;

forParam: id '∫' lBound ',' uBound '∫'
;

lBound:  number
| '-' number
| id
;

uBound:  number
| '-' number
| id
;

e: expression
;

OPERATION: '+'
| '-'
| '*'
| '/'
;

conditionExpression: condition
| condition conditionOp condition
|
;

conditionOp: '&&'
| '||'
;

condition: expression comparison expression
;

comparison: '=='
| '!='
| '>'
| '>='
| '<'
| '<='
;  

subscript: '$' id '$' subscript
|
;

/*expression: term
| term ADDOP expression
;*/

expression: term expressionPrime;

expressionPrime: addOp expression
|
;

addOp: '+'
| '-'
;

term: factor
| term mulOp factor
;

mulOp: '*'
| '/'
;

factor: id
| id '[' expression ']'
| id '(' exprList ')'
| id '[' exprList ']'
| number
| '-' number
| '(' expression ')'
| id subscript
| id subscript '[' expression ']'
| id subscript '(' exprList ')'
| id subscript '[' exprList ']'
;

exprList: ' '
| nonEmptyExprList
;

nonEmptyExprList: expression
| nonEmptyExprList ',' expression
;

r: id
| id '[' exprList ']'
| id subscript
| id subscript '[' exprList ']'
;

LETTER: [a-z]
| [A-Z]
| '_'
;

DIGIT: [0-9]+
;

id: LETTER (LETTER | DIGIT)*
;

number: DIGIT+
;

WS : [ \t\r\n]+ -> skip
;