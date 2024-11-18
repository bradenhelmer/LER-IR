// Generated from Glory.g4 by ANTLR 4.13.1
import org.antlr.v4.runtime.tree.ParseTreeListener;

/**
 * This interface defines a complete listener for a parse tree produced by
 * {@link GloryParser}.
 */
public interface GloryListener extends ParseTreeListener {
	/**
	 * Enter a parse tree produced by {@link GloryParser#statement}.
	 * @param ctx the parse tree
	 */
	void enterStatement(GloryParser.StatementContext ctx);
	/**
	 * Exit a parse tree produced by {@link GloryParser#statement}.
	 * @param ctx the parse tree
	 */
	void exitStatement(GloryParser.StatementContext ctx);
	/**
	 * Enter a parse tree produced by {@link GloryParser#l}.
	 * @param ctx the parse tree
	 */
	void enterL(GloryParser.LContext ctx);
	/**
	 * Exit a parse tree produced by {@link GloryParser#l}.
	 * @param ctx the parse tree
	 */
	void exitL(GloryParser.LContext ctx);
	/**
	 * Enter a parse tree produced by {@link GloryParser#forParam}.
	 * @param ctx the parse tree
	 */
	void enterForParam(GloryParser.ForParamContext ctx);
	/**
	 * Exit a parse tree produced by {@link GloryParser#forParam}.
	 * @param ctx the parse tree
	 */
	void exitForParam(GloryParser.ForParamContext ctx);
	/**
	 * Enter a parse tree produced by {@link GloryParser#lBound}.
	 * @param ctx the parse tree
	 */
	void enterLBound(GloryParser.LBoundContext ctx);
	/**
	 * Exit a parse tree produced by {@link GloryParser#lBound}.
	 * @param ctx the parse tree
	 */
	void exitLBound(GloryParser.LBoundContext ctx);
	/**
	 * Enter a parse tree produced by {@link GloryParser#uBound}.
	 * @param ctx the parse tree
	 */
	void enterUBound(GloryParser.UBoundContext ctx);
	/**
	 * Exit a parse tree produced by {@link GloryParser#uBound}.
	 * @param ctx the parse tree
	 */
	void exitUBound(GloryParser.UBoundContext ctx);
	/**
	 * Enter a parse tree produced by {@link GloryParser#e}.
	 * @param ctx the parse tree
	 */
	void enterE(GloryParser.EContext ctx);
	/**
	 * Exit a parse tree produced by {@link GloryParser#e}.
	 * @param ctx the parse tree
	 */
	void exitE(GloryParser.EContext ctx);
	/**
	 * Enter a parse tree produced by {@link GloryParser#conditionExpression}.
	 * @param ctx the parse tree
	 */
	void enterConditionExpression(GloryParser.ConditionExpressionContext ctx);
	/**
	 * Exit a parse tree produced by {@link GloryParser#conditionExpression}.
	 * @param ctx the parse tree
	 */
	void exitConditionExpression(GloryParser.ConditionExpressionContext ctx);
	/**
	 * Enter a parse tree produced by {@link GloryParser#conditionOp}.
	 * @param ctx the parse tree
	 */
	void enterConditionOp(GloryParser.ConditionOpContext ctx);
	/**
	 * Exit a parse tree produced by {@link GloryParser#conditionOp}.
	 * @param ctx the parse tree
	 */
	void exitConditionOp(GloryParser.ConditionOpContext ctx);
	/**
	 * Enter a parse tree produced by {@link GloryParser#condition}.
	 * @param ctx the parse tree
	 */
	void enterCondition(GloryParser.ConditionContext ctx);
	/**
	 * Exit a parse tree produced by {@link GloryParser#condition}.
	 * @param ctx the parse tree
	 */
	void exitCondition(GloryParser.ConditionContext ctx);
	/**
	 * Enter a parse tree produced by {@link GloryParser#comparison}.
	 * @param ctx the parse tree
	 */
	void enterComparison(GloryParser.ComparisonContext ctx);
	/**
	 * Exit a parse tree produced by {@link GloryParser#comparison}.
	 * @param ctx the parse tree
	 */
	void exitComparison(GloryParser.ComparisonContext ctx);
	/**
	 * Enter a parse tree produced by {@link GloryParser#subscript}.
	 * @param ctx the parse tree
	 */
	void enterSubscript(GloryParser.SubscriptContext ctx);
	/**
	 * Exit a parse tree produced by {@link GloryParser#subscript}.
	 * @param ctx the parse tree
	 */
	void exitSubscript(GloryParser.SubscriptContext ctx);
	/**
	 * Enter a parse tree produced by {@link GloryParser#expression}.
	 * @param ctx the parse tree
	 */
	void enterExpression(GloryParser.ExpressionContext ctx);
	/**
	 * Exit a parse tree produced by {@link GloryParser#expression}.
	 * @param ctx the parse tree
	 */
	void exitExpression(GloryParser.ExpressionContext ctx);
	/**
	 * Enter a parse tree produced by {@link GloryParser#expressionPrime}.
	 * @param ctx the parse tree
	 */
	void enterExpressionPrime(GloryParser.ExpressionPrimeContext ctx);
	/**
	 * Exit a parse tree produced by {@link GloryParser#expressionPrime}.
	 * @param ctx the parse tree
	 */
	void exitExpressionPrime(GloryParser.ExpressionPrimeContext ctx);
	/**
	 * Enter a parse tree produced by {@link GloryParser#addOp}.
	 * @param ctx the parse tree
	 */
	void enterAddOp(GloryParser.AddOpContext ctx);
	/**
	 * Exit a parse tree produced by {@link GloryParser#addOp}.
	 * @param ctx the parse tree
	 */
	void exitAddOp(GloryParser.AddOpContext ctx);
	/**
	 * Enter a parse tree produced by {@link GloryParser#term}.
	 * @param ctx the parse tree
	 */
	void enterTerm(GloryParser.TermContext ctx);
	/**
	 * Exit a parse tree produced by {@link GloryParser#term}.
	 * @param ctx the parse tree
	 */
	void exitTerm(GloryParser.TermContext ctx);
	/**
	 * Enter a parse tree produced by {@link GloryParser#mulOp}.
	 * @param ctx the parse tree
	 */
	void enterMulOp(GloryParser.MulOpContext ctx);
	/**
	 * Exit a parse tree produced by {@link GloryParser#mulOp}.
	 * @param ctx the parse tree
	 */
	void exitMulOp(GloryParser.MulOpContext ctx);
	/**
	 * Enter a parse tree produced by {@link GloryParser#factor}.
	 * @param ctx the parse tree
	 */
	void enterFactor(GloryParser.FactorContext ctx);
	/**
	 * Exit a parse tree produced by {@link GloryParser#factor}.
	 * @param ctx the parse tree
	 */
	void exitFactor(GloryParser.FactorContext ctx);
	/**
	 * Enter a parse tree produced by {@link GloryParser#exprList}.
	 * @param ctx the parse tree
	 */
	void enterExprList(GloryParser.ExprListContext ctx);
	/**
	 * Exit a parse tree produced by {@link GloryParser#exprList}.
	 * @param ctx the parse tree
	 */
	void exitExprList(GloryParser.ExprListContext ctx);
	/**
	 * Enter a parse tree produced by {@link GloryParser#nonEmptyExprList}.
	 * @param ctx the parse tree
	 */
	void enterNonEmptyExprList(GloryParser.NonEmptyExprListContext ctx);
	/**
	 * Exit a parse tree produced by {@link GloryParser#nonEmptyExprList}.
	 * @param ctx the parse tree
	 */
	void exitNonEmptyExprList(GloryParser.NonEmptyExprListContext ctx);
	/**
	 * Enter a parse tree produced by {@link GloryParser#r}.
	 * @param ctx the parse tree
	 */
	void enterR(GloryParser.RContext ctx);
	/**
	 * Exit a parse tree produced by {@link GloryParser#r}.
	 * @param ctx the parse tree
	 */
	void exitR(GloryParser.RContext ctx);
	/**
	 * Enter a parse tree produced by {@link GloryParser#id}.
	 * @param ctx the parse tree
	 */
	void enterId(GloryParser.IdContext ctx);
	/**
	 * Exit a parse tree produced by {@link GloryParser#id}.
	 * @param ctx the parse tree
	 */
	void exitId(GloryParser.IdContext ctx);
	/**
	 * Enter a parse tree produced by {@link GloryParser#number}.
	 * @param ctx the parse tree
	 */
	void enterNumber(GloryParser.NumberContext ctx);
	/**
	 * Exit a parse tree produced by {@link GloryParser#number}.
	 * @param ctx the parse tree
	 */
	void exitNumber(GloryParser.NumberContext ctx);
}