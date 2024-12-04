package optimizer;

// Generated from Glory.g4 by ANTLR 4.13.1
import org.antlr.v4.runtime.atn.*;
import org.antlr.v4.runtime.dfa.DFA;
import org.antlr.v4.runtime.*;
import org.antlr.v4.runtime.misc.*;
import org.antlr.v4.runtime.tree.*;
import java.util.List;
import java.util.Iterator;
import java.util.ArrayList;

@SuppressWarnings({"all", "warnings", "unchecked", "unused", "cast", "CheckReturnValue"})
public class GloryParser extends Parser {
	static { RuntimeMetaData.checkVersion("4.13.1", RuntimeMetaData.VERSION); }

	protected static final DFA[] _decisionToDFA;
	protected static final PredictionContextCache _sharedContextCache =
		new PredictionContextCache();
	public static final int
		T__0=1, T__1=2, T__2=3, T__3=4, T__4=5, T__5=6, T__6=7, T__7=8, T__8=9, 
		T__9=10, T__10=11, T__11=12, T__12=13, T__13=14, T__14=15, T__15=16, T__16=17, 
		T__17=18, T__18=19, T__19=20, T__20=21, NORMAL=22, SUMMATION=23, PRODUCT=24, 
		OTHER=25, OPERATION=26, LETTER=27, DIGIT=28, WS=29;
	public static final int
		RULE_statement = 0, RULE_l = 1, RULE_forParam = 2, RULE_lBound = 3, RULE_uBound = 4, 
		RULE_e = 5, RULE_conditionExpression = 6, RULE_conditionOp = 7, RULE_condition = 8, 
		RULE_comparison = 9, RULE_subscript = 10, RULE_expression = 11, RULE_expressionPrime = 12, 
		RULE_addOp = 13, RULE_term = 14, RULE_mulOp = 15, RULE_factor = 16, RULE_exprList = 17, 
		RULE_nonEmptyExprList = 18, RULE_r = 19, RULE_id = 20, RULE_number = 21;
	private static String[] makeRuleNames() {
		return new String[] {
			"statement", "l", "forParam", "lBound", "uBound", "e", "conditionExpression", 
			"conditionOp", "condition", "comparison", "subscript", "expression", 
			"expressionPrime", "addOp", "term", "mulOp", "factor", "exprList", "nonEmptyExprList", 
			"r", "id", "number"
		};
	}
	public static final String[] ruleNames = makeRuleNames();

	private static String[] makeLiteralNames() {
		return new String[] {
			null, "'='", "'\\u222B'", "','", "'-'", "'&&'", "'||'", "'=='", "'!='", 
			"'>'", "'>='", "'<'", "'<='", "'$'", "'+'", "'*'", "'/'", "'['", "']'", 
			"'('", "')'", "' '", "'\\u0393'", "'\\u03A3'", "'\\u03A0'", "'\\u03A8'"
		};
	}
	private static final String[] _LITERAL_NAMES = makeLiteralNames();
	private static String[] makeSymbolicNames() {
		return new String[] {
			null, null, null, null, null, null, null, null, null, null, null, null, 
			null, null, null, null, null, null, null, null, null, null, "NORMAL", 
			"SUMMATION", "PRODUCT", "OTHER", "OPERATION", "LETTER", "DIGIT", "WS"
		};
	}
	private static final String[] _SYMBOLIC_NAMES = makeSymbolicNames();
	public static final Vocabulary VOCABULARY = new VocabularyImpl(_LITERAL_NAMES, _SYMBOLIC_NAMES);

	/**
	 * @deprecated Use {@link #VOCABULARY} instead.
	 */
	@Deprecated
	public static final String[] tokenNames;
	static {
		tokenNames = new String[_SYMBOLIC_NAMES.length];
		for (int i = 0; i < tokenNames.length; i++) {
			tokenNames[i] = VOCABULARY.getLiteralName(i);
			if (tokenNames[i] == null) {
				tokenNames[i] = VOCABULARY.getSymbolicName(i);
			}

			if (tokenNames[i] == null) {
				tokenNames[i] = "<INVALID>";
			}
		}
	}

	@Override
	@Deprecated
	public String[] getTokenNames() {
		return tokenNames;
	}

	@Override

	public Vocabulary getVocabulary() {
		return VOCABULARY;
	}

	@Override
	public String getGrammarFileName() { return "Glory.g4"; }

	@Override
	public String[] getRuleNames() { return ruleNames; }

	@Override
	public String getSerializedATN() { return _serializedATN; }

	@Override
	public ATN getATN() { return _ATN; }

	public GloryParser(TokenStream input) {
		super(input);
		_interp = new ParserATNSimulator(this,_ATN,_decisionToDFA,_sharedContextCache);
	}

	@SuppressWarnings("CheckReturnValue")
	public static class StatementContext extends ParserRuleContext {
		public LContext l() {
			return getRuleContext(LContext.class,0);
		}
		public EContext e() {
			return getRuleContext(EContext.class,0);
		}
		public RContext r() {
			return getRuleContext(RContext.class,0);
		}
		public StatementContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_statement; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof GloryListener ) ((GloryListener)listener).enterStatement(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof GloryListener ) ((GloryListener)listener).exitStatement(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof GloryVisitor ) return ((GloryVisitor<? extends T>)visitor).visitStatement(this);
			else return visitor.visitChildren(this);
		}
	}

	public final StatementContext statement() throws RecognitionException {
		StatementContext _localctx = new StatementContext(_ctx, getState());
		enterRule(_localctx, 0, RULE_statement);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(44);
			l();
			setState(45);
			e();
			setState(46);
			match(T__0);
			setState(47);
			r();
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class LContext extends ParserRuleContext {
		public Token loopType;
		public ForParamContext forParam() {
			return getRuleContext(ForParamContext.class,0);
		}
		public LContext l() {
			return getRuleContext(LContext.class,0);
		}
		public TerminalNode NORMAL() { return getToken(GloryParser.NORMAL, 0); }
		public TerminalNode SUMMATION() { return getToken(GloryParser.SUMMATION, 0); }
		public TerminalNode PRODUCT() { return getToken(GloryParser.PRODUCT, 0); }
		public SubscriptContext subscript() {
			return getRuleContext(SubscriptContext.class,0);
		}
		public ConditionExpressionContext conditionExpression() {
			return getRuleContext(ConditionExpressionContext.class,0);
		}
		public TerminalNode OTHER() { return getToken(GloryParser.OTHER, 0); }
		public LContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_l; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof GloryListener ) ((GloryListener)listener).enterL(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof GloryListener ) ((GloryListener)listener).exitL(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof GloryVisitor ) return ((GloryVisitor<? extends T>)visitor).visitL(this);
			else return visitor.visitChildren(this);
		}
	}

	public final LContext l() throws RecognitionException {
		LContext _localctx = new LContext(_ctx, getState());
		enterRule(_localctx, 2, RULE_l);
		int _la;
		try {
			setState(60);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case NORMAL:
			case SUMMATION:
			case PRODUCT:
				enterOuterAlt(_localctx, 1);
				{
				setState(49);
				((LContext)_localctx).loopType = _input.LT(1);
				_la = _input.LA(1);
				if ( !((((_la) & ~0x3f) == 0 && ((1L << _la) & 29360128L) != 0)) ) {
					((LContext)_localctx).loopType = (Token)_errHandler.recoverInline(this);
				}
				else {
					if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
					_errHandler.reportMatch(this);
					consume();
				}
				setState(50);
				forParam();
				setState(51);
				l();
				}
				break;
			case OTHER:
				enterOuterAlt(_localctx, 2);
				{
				setState(53);
				((LContext)_localctx).loopType = match(OTHER);
				setState(54);
				subscript();
				setState(55);
				conditionExpression();
				setState(56);
				match(T__1);
				setState(57);
				l();
				}
				break;
			case T__3:
			case T__18:
			case LETTER:
			case DIGIT:
				enterOuterAlt(_localctx, 3);
				{
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class ForParamContext extends ParserRuleContext {
		public IdContext id() {
			return getRuleContext(IdContext.class,0);
		}
		public LBoundContext lBound() {
			return getRuleContext(LBoundContext.class,0);
		}
		public UBoundContext uBound() {
			return getRuleContext(UBoundContext.class,0);
		}
		public ForParamContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_forParam; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof GloryListener ) ((GloryListener)listener).enterForParam(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof GloryListener ) ((GloryListener)listener).exitForParam(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof GloryVisitor ) return ((GloryVisitor<? extends T>)visitor).visitForParam(this);
			else return visitor.visitChildren(this);
		}
	}

	public final ForParamContext forParam() throws RecognitionException {
		ForParamContext _localctx = new ForParamContext(_ctx, getState());
		enterRule(_localctx, 4, RULE_forParam);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(62);
			id();
			setState(63);
			match(T__1);
			setState(64);
			lBound();
			setState(65);
			match(T__2);
			setState(66);
			uBound();
			setState(67);
			match(T__1);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class LBoundContext extends ParserRuleContext {
		public NumberContext number() {
			return getRuleContext(NumberContext.class,0);
		}
		public IdContext id() {
			return getRuleContext(IdContext.class,0);
		}
		public LBoundContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_lBound; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof GloryListener ) ((GloryListener)listener).enterLBound(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof GloryListener ) ((GloryListener)listener).exitLBound(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof GloryVisitor ) return ((GloryVisitor<? extends T>)visitor).visitLBound(this);
			else return visitor.visitChildren(this);
		}
	}

	public final LBoundContext lBound() throws RecognitionException {
		LBoundContext _localctx = new LBoundContext(_ctx, getState());
		enterRule(_localctx, 6, RULE_lBound);
		try {
			setState(73);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case DIGIT:
				enterOuterAlt(_localctx, 1);
				{
				setState(69);
				number();
				}
				break;
			case T__3:
				enterOuterAlt(_localctx, 2);
				{
				setState(70);
				match(T__3);
				setState(71);
				number();
				}
				break;
			case LETTER:
				enterOuterAlt(_localctx, 3);
				{
				setState(72);
				id();
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class UBoundContext extends ParserRuleContext {
		public NumberContext number() {
			return getRuleContext(NumberContext.class,0);
		}
		public IdContext id() {
			return getRuleContext(IdContext.class,0);
		}
		public UBoundContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_uBound; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof GloryListener ) ((GloryListener)listener).enterUBound(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof GloryListener ) ((GloryListener)listener).exitUBound(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof GloryVisitor ) return ((GloryVisitor<? extends T>)visitor).visitUBound(this);
			else return visitor.visitChildren(this);
		}
	}

	public final UBoundContext uBound() throws RecognitionException {
		UBoundContext _localctx = new UBoundContext(_ctx, getState());
		enterRule(_localctx, 8, RULE_uBound);
		try {
			setState(79);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case DIGIT:
				enterOuterAlt(_localctx, 1);
				{
				setState(75);
				number();
				}
				break;
			case T__3:
				enterOuterAlt(_localctx, 2);
				{
				setState(76);
				match(T__3);
				setState(77);
				number();
				}
				break;
			case LETTER:
				enterOuterAlt(_localctx, 3);
				{
				setState(78);
				id();
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class EContext extends ParserRuleContext {
		public ExpressionContext expression() {
			return getRuleContext(ExpressionContext.class,0);
		}
		public EContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_e; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof GloryListener ) ((GloryListener)listener).enterE(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof GloryListener ) ((GloryListener)listener).exitE(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof GloryVisitor ) return ((GloryVisitor<? extends T>)visitor).visitE(this);
			else return visitor.visitChildren(this);
		}
	}

	public final EContext e() throws RecognitionException {
		EContext _localctx = new EContext(_ctx, getState());
		enterRule(_localctx, 10, RULE_e);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(81);
			expression();
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class ConditionExpressionContext extends ParserRuleContext {
		public List<ConditionContext> condition() {
			return getRuleContexts(ConditionContext.class);
		}
		public ConditionContext condition(int i) {
			return getRuleContext(ConditionContext.class,i);
		}
		public ConditionOpContext conditionOp() {
			return getRuleContext(ConditionOpContext.class,0);
		}
		public ConditionExpressionContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_conditionExpression; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof GloryListener ) ((GloryListener)listener).enterConditionExpression(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof GloryListener ) ((GloryListener)listener).exitConditionExpression(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof GloryVisitor ) return ((GloryVisitor<? extends T>)visitor).visitConditionExpression(this);
			else return visitor.visitChildren(this);
		}
	}

	public final ConditionExpressionContext conditionExpression() throws RecognitionException {
		ConditionExpressionContext _localctx = new ConditionExpressionContext(_ctx, getState());
		enterRule(_localctx, 12, RULE_conditionExpression);
		try {
			setState(89);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,3,_ctx) ) {
			case 1:
				enterOuterAlt(_localctx, 1);
				{
				setState(83);
				condition();
				}
				break;
			case 2:
				enterOuterAlt(_localctx, 2);
				{
				setState(84);
				condition();
				setState(85);
				conditionOp();
				setState(86);
				condition();
				}
				break;
			case 3:
				enterOuterAlt(_localctx, 3);
				{
				}
				break;
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class ConditionOpContext extends ParserRuleContext {
		public ConditionOpContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_conditionOp; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof GloryListener ) ((GloryListener)listener).enterConditionOp(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof GloryListener ) ((GloryListener)listener).exitConditionOp(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof GloryVisitor ) return ((GloryVisitor<? extends T>)visitor).visitConditionOp(this);
			else return visitor.visitChildren(this);
		}
	}

	public final ConditionOpContext conditionOp() throws RecognitionException {
		ConditionOpContext _localctx = new ConditionOpContext(_ctx, getState());
		enterRule(_localctx, 14, RULE_conditionOp);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(91);
			_la = _input.LA(1);
			if ( !(_la==T__4 || _la==T__5) ) {
			_errHandler.recoverInline(this);
			}
			else {
				if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
				_errHandler.reportMatch(this);
				consume();
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class ConditionContext extends ParserRuleContext {
		public List<ExpressionContext> expression() {
			return getRuleContexts(ExpressionContext.class);
		}
		public ExpressionContext expression(int i) {
			return getRuleContext(ExpressionContext.class,i);
		}
		public ComparisonContext comparison() {
			return getRuleContext(ComparisonContext.class,0);
		}
		public ConditionContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_condition; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof GloryListener ) ((GloryListener)listener).enterCondition(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof GloryListener ) ((GloryListener)listener).exitCondition(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof GloryVisitor ) return ((GloryVisitor<? extends T>)visitor).visitCondition(this);
			else return visitor.visitChildren(this);
		}
	}

	public final ConditionContext condition() throws RecognitionException {
		ConditionContext _localctx = new ConditionContext(_ctx, getState());
		enterRule(_localctx, 16, RULE_condition);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(93);
			expression();
			setState(94);
			comparison();
			setState(95);
			expression();
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class ComparisonContext extends ParserRuleContext {
		public ComparisonContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_comparison; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof GloryListener ) ((GloryListener)listener).enterComparison(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof GloryListener ) ((GloryListener)listener).exitComparison(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof GloryVisitor ) return ((GloryVisitor<? extends T>)visitor).visitComparison(this);
			else return visitor.visitChildren(this);
		}
	}

	public final ComparisonContext comparison() throws RecognitionException {
		ComparisonContext _localctx = new ComparisonContext(_ctx, getState());
		enterRule(_localctx, 18, RULE_comparison);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(97);
			_la = _input.LA(1);
			if ( !((((_la) & ~0x3f) == 0 && ((1L << _la) & 8064L) != 0)) ) {
			_errHandler.recoverInline(this);
			}
			else {
				if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
				_errHandler.reportMatch(this);
				consume();
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class SubscriptContext extends ParserRuleContext {
		public IdContext id() {
			return getRuleContext(IdContext.class,0);
		}
		public SubscriptContext subscript() {
			return getRuleContext(SubscriptContext.class,0);
		}
		public SubscriptContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_subscript; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof GloryListener ) ((GloryListener)listener).enterSubscript(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof GloryListener ) ((GloryListener)listener).exitSubscript(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof GloryVisitor ) return ((GloryVisitor<? extends T>)visitor).visitSubscript(this);
			else return visitor.visitChildren(this);
		}
	}

	public final SubscriptContext subscript() throws RecognitionException {
		SubscriptContext _localctx = new SubscriptContext(_ctx, getState());
		enterRule(_localctx, 20, RULE_subscript);
		try {
			setState(105);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,4,_ctx) ) {
			case 1:
				enterOuterAlt(_localctx, 1);
				{
				setState(99);
				match(T__12);
				setState(100);
				id();
				setState(101);
				match(T__12);
				setState(102);
				subscript();
				}
				break;
			case 2:
				enterOuterAlt(_localctx, 2);
				{
				}
				break;
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class ExpressionContext extends ParserRuleContext {
		public TermContext term() {
			return getRuleContext(TermContext.class,0);
		}
		public ExpressionPrimeContext expressionPrime() {
			return getRuleContext(ExpressionPrimeContext.class,0);
		}
		public ExpressionContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_expression; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof GloryListener ) ((GloryListener)listener).enterExpression(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof GloryListener ) ((GloryListener)listener).exitExpression(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof GloryVisitor ) return ((GloryVisitor<? extends T>)visitor).visitExpression(this);
			else return visitor.visitChildren(this);
		}
	}

	public final ExpressionContext expression() throws RecognitionException {
		ExpressionContext _localctx = new ExpressionContext(_ctx, getState());
		enterRule(_localctx, 22, RULE_expression);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(107);
			term(0);
			setState(108);
			expressionPrime();
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class ExpressionPrimeContext extends ParserRuleContext {
		public AddOpContext addOp() {
			return getRuleContext(AddOpContext.class,0);
		}
		public ExpressionContext expression() {
			return getRuleContext(ExpressionContext.class,0);
		}
		public ExpressionPrimeContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_expressionPrime; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof GloryListener ) ((GloryListener)listener).enterExpressionPrime(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof GloryListener ) ((GloryListener)listener).exitExpressionPrime(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof GloryVisitor ) return ((GloryVisitor<? extends T>)visitor).visitExpressionPrime(this);
			else return visitor.visitChildren(this);
		}
	}

	public final ExpressionPrimeContext expressionPrime() throws RecognitionException {
		ExpressionPrimeContext _localctx = new ExpressionPrimeContext(_ctx, getState());
		enterRule(_localctx, 24, RULE_expressionPrime);
		try {
			setState(114);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,5,_ctx) ) {
			case 1:
				enterOuterAlt(_localctx, 1);
				{
				setState(110);
				addOp();
				setState(111);
				expression();
				}
				break;
			case 2:
				enterOuterAlt(_localctx, 2);
				{
				}
				break;
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class AddOpContext extends ParserRuleContext {
		public AddOpContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_addOp; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof GloryListener ) ((GloryListener)listener).enterAddOp(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof GloryListener ) ((GloryListener)listener).exitAddOp(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof GloryVisitor ) return ((GloryVisitor<? extends T>)visitor).visitAddOp(this);
			else return visitor.visitChildren(this);
		}
	}

	public final AddOpContext addOp() throws RecognitionException {
		AddOpContext _localctx = new AddOpContext(_ctx, getState());
		enterRule(_localctx, 26, RULE_addOp);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(116);
			_la = _input.LA(1);
			if ( !(_la==T__3 || _la==T__13) ) {
			_errHandler.recoverInline(this);
			}
			else {
				if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
				_errHandler.reportMatch(this);
				consume();
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class TermContext extends ParserRuleContext {
		public FactorContext factor() {
			return getRuleContext(FactorContext.class,0);
		}
		public TermContext term() {
			return getRuleContext(TermContext.class,0);
		}
		public MulOpContext mulOp() {
			return getRuleContext(MulOpContext.class,0);
		}
		public TermContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_term; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof GloryListener ) ((GloryListener)listener).enterTerm(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof GloryListener ) ((GloryListener)listener).exitTerm(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof GloryVisitor ) return ((GloryVisitor<? extends T>)visitor).visitTerm(this);
			else return visitor.visitChildren(this);
		}
	}

	public final TermContext term() throws RecognitionException {
		return term(0);
	}

	private TermContext term(int _p) throws RecognitionException {
		ParserRuleContext _parentctx = _ctx;
		int _parentState = getState();
		TermContext _localctx = new TermContext(_ctx, _parentState);
		TermContext _prevctx = _localctx;
		int _startState = 28;
		enterRecursionRule(_localctx, 28, RULE_term, _p);
		try {
			int _alt;
			enterOuterAlt(_localctx, 1);
			{
			{
			setState(119);
			factor();
			}
			_ctx.stop = _input.LT(-1);
			setState(127);
			_errHandler.sync(this);
			_alt = getInterpreter().adaptivePredict(_input,6,_ctx);
			while ( _alt!=2 && _alt!=org.antlr.v4.runtime.atn.ATN.INVALID_ALT_NUMBER ) {
				if ( _alt==1 ) {
					if ( _parseListeners!=null ) triggerExitRuleEvent();
					_prevctx = _localctx;
					{
					{
					_localctx = new TermContext(_parentctx, _parentState);
					pushNewRecursionContext(_localctx, _startState, RULE_term);
					setState(121);
					if (!(precpred(_ctx, 1))) throw new FailedPredicateException(this, "precpred(_ctx, 1)");
					setState(122);
					mulOp();
					setState(123);
					factor();
					}
					} 
				}
				setState(129);
				_errHandler.sync(this);
				_alt = getInterpreter().adaptivePredict(_input,6,_ctx);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			unrollRecursionContexts(_parentctx);
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class MulOpContext extends ParserRuleContext {
		public MulOpContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_mulOp; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof GloryListener ) ((GloryListener)listener).enterMulOp(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof GloryListener ) ((GloryListener)listener).exitMulOp(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof GloryVisitor ) return ((GloryVisitor<? extends T>)visitor).visitMulOp(this);
			else return visitor.visitChildren(this);
		}
	}

	public final MulOpContext mulOp() throws RecognitionException {
		MulOpContext _localctx = new MulOpContext(_ctx, getState());
		enterRule(_localctx, 30, RULE_mulOp);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(130);
			_la = _input.LA(1);
			if ( !(_la==T__14 || _la==T__15) ) {
			_errHandler.recoverInline(this);
			}
			else {
				if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
				_errHandler.reportMatch(this);
				consume();
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class FactorContext extends ParserRuleContext {
		public IdContext id() {
			return getRuleContext(IdContext.class,0);
		}
		public ExpressionContext expression() {
			return getRuleContext(ExpressionContext.class,0);
		}
		public ExprListContext exprList() {
			return getRuleContext(ExprListContext.class,0);
		}
		public NumberContext number() {
			return getRuleContext(NumberContext.class,0);
		}
		public SubscriptContext subscript() {
			return getRuleContext(SubscriptContext.class,0);
		}
		public FactorContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_factor; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof GloryListener ) ((GloryListener)listener).enterFactor(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof GloryListener ) ((GloryListener)listener).exitFactor(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof GloryVisitor ) return ((GloryVisitor<? extends T>)visitor).visitFactor(this);
			else return visitor.visitChildren(this);
		}
	}

	public final FactorContext factor() throws RecognitionException {
		FactorContext _localctx = new FactorContext(_ctx, getState());
		enterRule(_localctx, 32, RULE_factor);
		try {
			setState(176);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,7,_ctx) ) {
			case 1:
				enterOuterAlt(_localctx, 1);
				{
				setState(132);
				id();
				}
				break;
			case 2:
				enterOuterAlt(_localctx, 2);
				{
				setState(133);
				id();
				setState(134);
				match(T__16);
				setState(135);
				expression();
				setState(136);
				match(T__17);
				}
				break;
			case 3:
				enterOuterAlt(_localctx, 3);
				{
				setState(138);
				id();
				setState(139);
				match(T__18);
				setState(140);
				exprList();
				setState(141);
				match(T__19);
				}
				break;
			case 4:
				enterOuterAlt(_localctx, 4);
				{
				setState(143);
				id();
				setState(144);
				match(T__16);
				setState(145);
				exprList();
				setState(146);
				match(T__17);
				}
				break;
			case 5:
				enterOuterAlt(_localctx, 5);
				{
				setState(148);
				number();
				}
				break;
			case 6:
				enterOuterAlt(_localctx, 6);
				{
				setState(149);
				match(T__3);
				setState(150);
				number();
				}
				break;
			case 7:
				enterOuterAlt(_localctx, 7);
				{
				setState(151);
				match(T__18);
				setState(152);
				expression();
				setState(153);
				match(T__19);
				}
				break;
			case 8:
				enterOuterAlt(_localctx, 8);
				{
				setState(155);
				id();
				setState(156);
				subscript();
				}
				break;
			case 9:
				enterOuterAlt(_localctx, 9);
				{
				setState(158);
				id();
				setState(159);
				subscript();
				setState(160);
				match(T__16);
				setState(161);
				expression();
				setState(162);
				match(T__17);
				}
				break;
			case 10:
				enterOuterAlt(_localctx, 10);
				{
				setState(164);
				id();
				setState(165);
				subscript();
				setState(166);
				match(T__18);
				setState(167);
				exprList();
				setState(168);
				match(T__19);
				}
				break;
			case 11:
				enterOuterAlt(_localctx, 11);
				{
				setState(170);
				id();
				setState(171);
				subscript();
				setState(172);
				match(T__16);
				setState(173);
				exprList();
				setState(174);
				match(T__17);
				}
				break;
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class ExprListContext extends ParserRuleContext {
		public NonEmptyExprListContext nonEmptyExprList() {
			return getRuleContext(NonEmptyExprListContext.class,0);
		}
		public ExprListContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_exprList; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof GloryListener ) ((GloryListener)listener).enterExprList(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof GloryListener ) ((GloryListener)listener).exitExprList(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof GloryVisitor ) return ((GloryVisitor<? extends T>)visitor).visitExprList(this);
			else return visitor.visitChildren(this);
		}
	}

	public final ExprListContext exprList() throws RecognitionException {
		ExprListContext _localctx = new ExprListContext(_ctx, getState());
		enterRule(_localctx, 34, RULE_exprList);
		try {
			setState(180);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case T__20:
				enterOuterAlt(_localctx, 1);
				{
				setState(178);
				match(T__20);
				}
				break;
			case T__3:
			case T__18:
			case LETTER:
			case DIGIT:
				enterOuterAlt(_localctx, 2);
				{
				setState(179);
				nonEmptyExprList(0);
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class NonEmptyExprListContext extends ParserRuleContext {
		public ExpressionContext expression() {
			return getRuleContext(ExpressionContext.class,0);
		}
		public NonEmptyExprListContext nonEmptyExprList() {
			return getRuleContext(NonEmptyExprListContext.class,0);
		}
		public NonEmptyExprListContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_nonEmptyExprList; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof GloryListener ) ((GloryListener)listener).enterNonEmptyExprList(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof GloryListener ) ((GloryListener)listener).exitNonEmptyExprList(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof GloryVisitor ) return ((GloryVisitor<? extends T>)visitor).visitNonEmptyExprList(this);
			else return visitor.visitChildren(this);
		}
	}

	public final NonEmptyExprListContext nonEmptyExprList() throws RecognitionException {
		return nonEmptyExprList(0);
	}

	private NonEmptyExprListContext nonEmptyExprList(int _p) throws RecognitionException {
		ParserRuleContext _parentctx = _ctx;
		int _parentState = getState();
		NonEmptyExprListContext _localctx = new NonEmptyExprListContext(_ctx, _parentState);
		NonEmptyExprListContext _prevctx = _localctx;
		int _startState = 36;
		enterRecursionRule(_localctx, 36, RULE_nonEmptyExprList, _p);
		try {
			int _alt;
			enterOuterAlt(_localctx, 1);
			{
			{
			setState(183);
			expression();
			}
			_ctx.stop = _input.LT(-1);
			setState(190);
			_errHandler.sync(this);
			_alt = getInterpreter().adaptivePredict(_input,9,_ctx);
			while ( _alt!=2 && _alt!=org.antlr.v4.runtime.atn.ATN.INVALID_ALT_NUMBER ) {
				if ( _alt==1 ) {
					if ( _parseListeners!=null ) triggerExitRuleEvent();
					_prevctx = _localctx;
					{
					{
					_localctx = new NonEmptyExprListContext(_parentctx, _parentState);
					pushNewRecursionContext(_localctx, _startState, RULE_nonEmptyExprList);
					setState(185);
					if (!(precpred(_ctx, 1))) throw new FailedPredicateException(this, "precpred(_ctx, 1)");
					setState(186);
					match(T__2);
					setState(187);
					expression();
					}
					} 
				}
				setState(192);
				_errHandler.sync(this);
				_alt = getInterpreter().adaptivePredict(_input,9,_ctx);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			unrollRecursionContexts(_parentctx);
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class RContext extends ParserRuleContext {
		public IdContext id() {
			return getRuleContext(IdContext.class,0);
		}
		public ExprListContext exprList() {
			return getRuleContext(ExprListContext.class,0);
		}
		public SubscriptContext subscript() {
			return getRuleContext(SubscriptContext.class,0);
		}
		public RContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_r; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof GloryListener ) ((GloryListener)listener).enterR(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof GloryListener ) ((GloryListener)listener).exitR(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof GloryVisitor ) return ((GloryVisitor<? extends T>)visitor).visitR(this);
			else return visitor.visitChildren(this);
		}
	}

	public final RContext r() throws RecognitionException {
		RContext _localctx = new RContext(_ctx, getState());
		enterRule(_localctx, 38, RULE_r);
		try {
			setState(208);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,10,_ctx) ) {
			case 1:
				enterOuterAlt(_localctx, 1);
				{
				setState(193);
				id();
				}
				break;
			case 2:
				enterOuterAlt(_localctx, 2);
				{
				setState(194);
				id();
				setState(195);
				match(T__16);
				setState(196);
				exprList();
				setState(197);
				match(T__17);
				}
				break;
			case 3:
				enterOuterAlt(_localctx, 3);
				{
				setState(199);
				id();
				setState(200);
				subscript();
				}
				break;
			case 4:
				enterOuterAlt(_localctx, 4);
				{
				setState(202);
				id();
				setState(203);
				subscript();
				setState(204);
				match(T__16);
				setState(205);
				exprList();
				setState(206);
				match(T__17);
				}
				break;
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class IdContext extends ParserRuleContext {
		public List<TerminalNode> LETTER() { return getTokens(GloryParser.LETTER); }
		public TerminalNode LETTER(int i) {
			return getToken(GloryParser.LETTER, i);
		}
		public List<TerminalNode> DIGIT() { return getTokens(GloryParser.DIGIT); }
		public TerminalNode DIGIT(int i) {
			return getToken(GloryParser.DIGIT, i);
		}
		public IdContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_id; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof GloryListener ) ((GloryListener)listener).enterId(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof GloryListener ) ((GloryListener)listener).exitId(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof GloryVisitor ) return ((GloryVisitor<? extends T>)visitor).visitId(this);
			else return visitor.visitChildren(this);
		}
	}

	public final IdContext id() throws RecognitionException {
		IdContext _localctx = new IdContext(_ctx, getState());
		enterRule(_localctx, 40, RULE_id);
		int _la;
		try {
			int _alt;
			enterOuterAlt(_localctx, 1);
			{
			setState(210);
			match(LETTER);
			setState(214);
			_errHandler.sync(this);
			_alt = getInterpreter().adaptivePredict(_input,11,_ctx);
			while ( _alt!=2 && _alt!=org.antlr.v4.runtime.atn.ATN.INVALID_ALT_NUMBER ) {
				if ( _alt==1 ) {
					{
					{
					setState(211);
					_la = _input.LA(1);
					if ( !(_la==LETTER || _la==DIGIT) ) {
					_errHandler.recoverInline(this);
					}
					else {
						if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
						_errHandler.reportMatch(this);
						consume();
					}
					}
					} 
				}
				setState(216);
				_errHandler.sync(this);
				_alt = getInterpreter().adaptivePredict(_input,11,_ctx);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class NumberContext extends ParserRuleContext {
		public List<TerminalNode> DIGIT() { return getTokens(GloryParser.DIGIT); }
		public TerminalNode DIGIT(int i) {
			return getToken(GloryParser.DIGIT, i);
		}
		public NumberContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_number; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof GloryListener ) ((GloryListener)listener).enterNumber(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof GloryListener ) ((GloryListener)listener).exitNumber(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof GloryVisitor ) return ((GloryVisitor<? extends T>)visitor).visitNumber(this);
			else return visitor.visitChildren(this);
		}
	}

	public final NumberContext number() throws RecognitionException {
		NumberContext _localctx = new NumberContext(_ctx, getState());
		enterRule(_localctx, 42, RULE_number);
		try {
			int _alt;
			enterOuterAlt(_localctx, 1);
			{
			setState(218); 
			_errHandler.sync(this);
			_alt = 1;
			do {
				switch (_alt) {
				case 1:
					{
					{
					setState(217);
					match(DIGIT);
					}
					}
					break;
				default:
					throw new NoViableAltException(this);
				}
				setState(220); 
				_errHandler.sync(this);
				_alt = getInterpreter().adaptivePredict(_input,12,_ctx);
			} while ( _alt!=2 && _alt!=org.antlr.v4.runtime.atn.ATN.INVALID_ALT_NUMBER );
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public boolean sempred(RuleContext _localctx, int ruleIndex, int predIndex) {
		switch (ruleIndex) {
		case 14:
			return term_sempred((TermContext)_localctx, predIndex);
		case 18:
			return nonEmptyExprList_sempred((NonEmptyExprListContext)_localctx, predIndex);
		}
		return true;
	}
	private boolean term_sempred(TermContext _localctx, int predIndex) {
		switch (predIndex) {
		case 0:
			return precpred(_ctx, 1);
		}
		return true;
	}
	private boolean nonEmptyExprList_sempred(NonEmptyExprListContext _localctx, int predIndex) {
		switch (predIndex) {
		case 1:
			return precpred(_ctx, 1);
		}
		return true;
	}

	public static final String _serializedATN =
		"\u0004\u0001\u001d\u00df\u0002\u0000\u0007\u0000\u0002\u0001\u0007\u0001"+
		"\u0002\u0002\u0007\u0002\u0002\u0003\u0007\u0003\u0002\u0004\u0007\u0004"+
		"\u0002\u0005\u0007\u0005\u0002\u0006\u0007\u0006\u0002\u0007\u0007\u0007"+
		"\u0002\b\u0007\b\u0002\t\u0007\t\u0002\n\u0007\n\u0002\u000b\u0007\u000b"+
		"\u0002\f\u0007\f\u0002\r\u0007\r\u0002\u000e\u0007\u000e\u0002\u000f\u0007"+
		"\u000f\u0002\u0010\u0007\u0010\u0002\u0011\u0007\u0011\u0002\u0012\u0007"+
		"\u0012\u0002\u0013\u0007\u0013\u0002\u0014\u0007\u0014\u0002\u0015\u0007"+
		"\u0015\u0001\u0000\u0001\u0000\u0001\u0000\u0001\u0000\u0001\u0000\u0001"+
		"\u0001\u0001\u0001\u0001\u0001\u0001\u0001\u0001\u0001\u0001\u0001\u0001"+
		"\u0001\u0001\u0001\u0001\u0001\u0001\u0001\u0001\u0001\u0003\u0001=\b"+
		"\u0001\u0001\u0002\u0001\u0002\u0001\u0002\u0001\u0002\u0001\u0002\u0001"+
		"\u0002\u0001\u0002\u0001\u0003\u0001\u0003\u0001\u0003\u0001\u0003\u0003"+
		"\u0003J\b\u0003\u0001\u0004\u0001\u0004\u0001\u0004\u0001\u0004\u0003"+
		"\u0004P\b\u0004\u0001\u0005\u0001\u0005\u0001\u0006\u0001\u0006\u0001"+
		"\u0006\u0001\u0006\u0001\u0006\u0001\u0006\u0003\u0006Z\b\u0006\u0001"+
		"\u0007\u0001\u0007\u0001\b\u0001\b\u0001\b\u0001\b\u0001\t\u0001\t\u0001"+
		"\n\u0001\n\u0001\n\u0001\n\u0001\n\u0001\n\u0003\nj\b\n\u0001\u000b\u0001"+
		"\u000b\u0001\u000b\u0001\f\u0001\f\u0001\f\u0001\f\u0003\fs\b\f\u0001"+
		"\r\u0001\r\u0001\u000e\u0001\u000e\u0001\u000e\u0001\u000e\u0001\u000e"+
		"\u0001\u000e\u0001\u000e\u0005\u000e~\b\u000e\n\u000e\f\u000e\u0081\t"+
		"\u000e\u0001\u000f\u0001\u000f\u0001\u0010\u0001\u0010\u0001\u0010\u0001"+
		"\u0010\u0001\u0010\u0001\u0010\u0001\u0010\u0001\u0010\u0001\u0010\u0001"+
		"\u0010\u0001\u0010\u0001\u0010\u0001\u0010\u0001\u0010\u0001\u0010\u0001"+
		"\u0010\u0001\u0010\u0001\u0010\u0001\u0010\u0001\u0010\u0001\u0010\u0001"+
		"\u0010\u0001\u0010\u0001\u0010\u0001\u0010\u0001\u0010\u0001\u0010\u0001"+
		"\u0010\u0001\u0010\u0001\u0010\u0001\u0010\u0001\u0010\u0001\u0010\u0001"+
		"\u0010\u0001\u0010\u0001\u0010\u0001\u0010\u0001\u0010\u0001\u0010\u0001"+
		"\u0010\u0001\u0010\u0001\u0010\u0001\u0010\u0001\u0010\u0003\u0010\u00b1"+
		"\b\u0010\u0001\u0011\u0001\u0011\u0003\u0011\u00b5\b\u0011\u0001\u0012"+
		"\u0001\u0012\u0001\u0012\u0001\u0012\u0001\u0012\u0001\u0012\u0005\u0012"+
		"\u00bd\b\u0012\n\u0012\f\u0012\u00c0\t\u0012\u0001\u0013\u0001\u0013\u0001"+
		"\u0013\u0001\u0013\u0001\u0013\u0001\u0013\u0001\u0013\u0001\u0013\u0001"+
		"\u0013\u0001\u0013\u0001\u0013\u0001\u0013\u0001\u0013\u0001\u0013\u0001"+
		"\u0013\u0003\u0013\u00d1\b\u0013\u0001\u0014\u0001\u0014\u0005\u0014\u00d5"+
		"\b\u0014\n\u0014\f\u0014\u00d8\t\u0014\u0001\u0015\u0004\u0015\u00db\b"+
		"\u0015\u000b\u0015\f\u0015\u00dc\u0001\u0015\u0000\u0002\u001c$\u0016"+
		"\u0000\u0002\u0004\u0006\b\n\f\u000e\u0010\u0012\u0014\u0016\u0018\u001a"+
		"\u001c\u001e \"$&(*\u0000\u0006\u0001\u0000\u0016\u0018\u0001\u0000\u0005"+
		"\u0006\u0001\u0000\u0007\f\u0002\u0000\u0004\u0004\u000e\u000e\u0001\u0000"+
		"\u000f\u0010\u0001\u0000\u001b\u001c\u00e4\u0000,\u0001\u0000\u0000\u0000"+
		"\u0002<\u0001\u0000\u0000\u0000\u0004>\u0001\u0000\u0000\u0000\u0006I"+
		"\u0001\u0000\u0000\u0000\bO\u0001\u0000\u0000\u0000\nQ\u0001\u0000\u0000"+
		"\u0000\fY\u0001\u0000\u0000\u0000\u000e[\u0001\u0000\u0000\u0000\u0010"+
		"]\u0001\u0000\u0000\u0000\u0012a\u0001\u0000\u0000\u0000\u0014i\u0001"+
		"\u0000\u0000\u0000\u0016k\u0001\u0000\u0000\u0000\u0018r\u0001\u0000\u0000"+
		"\u0000\u001at\u0001\u0000\u0000\u0000\u001cv\u0001\u0000\u0000\u0000\u001e"+
		"\u0082\u0001\u0000\u0000\u0000 \u00b0\u0001\u0000\u0000\u0000\"\u00b4"+
		"\u0001\u0000\u0000\u0000$\u00b6\u0001\u0000\u0000\u0000&\u00d0\u0001\u0000"+
		"\u0000\u0000(\u00d2\u0001\u0000\u0000\u0000*\u00da\u0001\u0000\u0000\u0000"+
		",-\u0003\u0002\u0001\u0000-.\u0003\n\u0005\u0000./\u0005\u0001\u0000\u0000"+
		"/0\u0003&\u0013\u00000\u0001\u0001\u0000\u0000\u000012\u0007\u0000\u0000"+
		"\u000023\u0003\u0004\u0002\u000034\u0003\u0002\u0001\u00004=\u0001\u0000"+
		"\u0000\u000056\u0005\u0019\u0000\u000067\u0003\u0014\n\u000078\u0003\f"+
		"\u0006\u000089\u0005\u0002\u0000\u00009:\u0003\u0002\u0001\u0000:=\u0001"+
		"\u0000\u0000\u0000;=\u0001\u0000\u0000\u0000<1\u0001\u0000\u0000\u0000"+
		"<5\u0001\u0000\u0000\u0000<;\u0001\u0000\u0000\u0000=\u0003\u0001\u0000"+
		"\u0000\u0000>?\u0003(\u0014\u0000?@\u0005\u0002\u0000\u0000@A\u0003\u0006"+
		"\u0003\u0000AB\u0005\u0003\u0000\u0000BC\u0003\b\u0004\u0000CD\u0005\u0002"+
		"\u0000\u0000D\u0005\u0001\u0000\u0000\u0000EJ\u0003*\u0015\u0000FG\u0005"+
		"\u0004\u0000\u0000GJ\u0003*\u0015\u0000HJ\u0003(\u0014\u0000IE\u0001\u0000"+
		"\u0000\u0000IF\u0001\u0000\u0000\u0000IH\u0001\u0000\u0000\u0000J\u0007"+
		"\u0001\u0000\u0000\u0000KP\u0003*\u0015\u0000LM\u0005\u0004\u0000\u0000"+
		"MP\u0003*\u0015\u0000NP\u0003(\u0014\u0000OK\u0001\u0000\u0000\u0000O"+
		"L\u0001\u0000\u0000\u0000ON\u0001\u0000\u0000\u0000P\t\u0001\u0000\u0000"+
		"\u0000QR\u0003\u0016\u000b\u0000R\u000b\u0001\u0000\u0000\u0000SZ\u0003"+
		"\u0010\b\u0000TU\u0003\u0010\b\u0000UV\u0003\u000e\u0007\u0000VW\u0003"+
		"\u0010\b\u0000WZ\u0001\u0000\u0000\u0000XZ\u0001\u0000\u0000\u0000YS\u0001"+
		"\u0000\u0000\u0000YT\u0001\u0000\u0000\u0000YX\u0001\u0000\u0000\u0000"+
		"Z\r\u0001\u0000\u0000\u0000[\\\u0007\u0001\u0000\u0000\\\u000f\u0001\u0000"+
		"\u0000\u0000]^\u0003\u0016\u000b\u0000^_\u0003\u0012\t\u0000_`\u0003\u0016"+
		"\u000b\u0000`\u0011\u0001\u0000\u0000\u0000ab\u0007\u0002\u0000\u0000"+
		"b\u0013\u0001\u0000\u0000\u0000cd\u0005\r\u0000\u0000de\u0003(\u0014\u0000"+
		"ef\u0005\r\u0000\u0000fg\u0003\u0014\n\u0000gj\u0001\u0000\u0000\u0000"+
		"hj\u0001\u0000\u0000\u0000ic\u0001\u0000\u0000\u0000ih\u0001\u0000\u0000"+
		"\u0000j\u0015\u0001\u0000\u0000\u0000kl\u0003\u001c\u000e\u0000lm\u0003"+
		"\u0018\f\u0000m\u0017\u0001\u0000\u0000\u0000no\u0003\u001a\r\u0000op"+
		"\u0003\u0016\u000b\u0000ps\u0001\u0000\u0000\u0000qs\u0001\u0000\u0000"+
		"\u0000rn\u0001\u0000\u0000\u0000rq\u0001\u0000\u0000\u0000s\u0019\u0001"+
		"\u0000\u0000\u0000tu\u0007\u0003\u0000\u0000u\u001b\u0001\u0000\u0000"+
		"\u0000vw\u0006\u000e\uffff\uffff\u0000wx\u0003 \u0010\u0000x\u007f\u0001"+
		"\u0000\u0000\u0000yz\n\u0001\u0000\u0000z{\u0003\u001e\u000f\u0000{|\u0003"+
		" \u0010\u0000|~\u0001\u0000\u0000\u0000}y\u0001\u0000\u0000\u0000~\u0081"+
		"\u0001\u0000\u0000\u0000\u007f}\u0001\u0000\u0000\u0000\u007f\u0080\u0001"+
		"\u0000\u0000\u0000\u0080\u001d\u0001\u0000\u0000\u0000\u0081\u007f\u0001"+
		"\u0000\u0000\u0000\u0082\u0083\u0007\u0004\u0000\u0000\u0083\u001f\u0001"+
		"\u0000\u0000\u0000\u0084\u00b1\u0003(\u0014\u0000\u0085\u0086\u0003(\u0014"+
		"\u0000\u0086\u0087\u0005\u0011\u0000\u0000\u0087\u0088\u0003\u0016\u000b"+
		"\u0000\u0088\u0089\u0005\u0012\u0000\u0000\u0089\u00b1\u0001\u0000\u0000"+
		"\u0000\u008a\u008b\u0003(\u0014\u0000\u008b\u008c\u0005\u0013\u0000\u0000"+
		"\u008c\u008d\u0003\"\u0011\u0000\u008d\u008e\u0005\u0014\u0000\u0000\u008e"+
		"\u00b1\u0001\u0000\u0000\u0000\u008f\u0090\u0003(\u0014\u0000\u0090\u0091"+
		"\u0005\u0011\u0000\u0000\u0091\u0092\u0003\"\u0011\u0000\u0092\u0093\u0005"+
		"\u0012\u0000\u0000\u0093\u00b1\u0001\u0000\u0000\u0000\u0094\u00b1\u0003"+
		"*\u0015\u0000\u0095\u0096\u0005\u0004\u0000\u0000\u0096\u00b1\u0003*\u0015"+
		"\u0000\u0097\u0098\u0005\u0013\u0000\u0000\u0098\u0099\u0003\u0016\u000b"+
		"\u0000\u0099\u009a\u0005\u0014\u0000\u0000\u009a\u00b1\u0001\u0000\u0000"+
		"\u0000\u009b\u009c\u0003(\u0014\u0000\u009c\u009d\u0003\u0014\n\u0000"+
		"\u009d\u00b1\u0001\u0000\u0000\u0000\u009e\u009f\u0003(\u0014\u0000\u009f"+
		"\u00a0\u0003\u0014\n\u0000\u00a0\u00a1\u0005\u0011\u0000\u0000\u00a1\u00a2"+
		"\u0003\u0016\u000b\u0000\u00a2\u00a3\u0005\u0012\u0000\u0000\u00a3\u00b1"+
		"\u0001\u0000\u0000\u0000\u00a4\u00a5\u0003(\u0014\u0000\u00a5\u00a6\u0003"+
		"\u0014\n\u0000\u00a6\u00a7\u0005\u0013\u0000\u0000\u00a7\u00a8\u0003\""+
		"\u0011\u0000\u00a8\u00a9\u0005\u0014\u0000\u0000\u00a9\u00b1\u0001\u0000"+
		"\u0000\u0000\u00aa\u00ab\u0003(\u0014\u0000\u00ab\u00ac\u0003\u0014\n"+
		"\u0000\u00ac\u00ad\u0005\u0011\u0000\u0000\u00ad\u00ae\u0003\"\u0011\u0000"+
		"\u00ae\u00af\u0005\u0012\u0000\u0000\u00af\u00b1\u0001\u0000\u0000\u0000"+
		"\u00b0\u0084\u0001\u0000\u0000\u0000\u00b0\u0085\u0001\u0000\u0000\u0000"+
		"\u00b0\u008a\u0001\u0000\u0000\u0000\u00b0\u008f\u0001\u0000\u0000\u0000"+
		"\u00b0\u0094\u0001\u0000\u0000\u0000\u00b0\u0095\u0001\u0000\u0000\u0000"+
		"\u00b0\u0097\u0001\u0000\u0000\u0000\u00b0\u009b\u0001\u0000\u0000\u0000"+
		"\u00b0\u009e\u0001\u0000\u0000\u0000\u00b0\u00a4\u0001\u0000\u0000\u0000"+
		"\u00b0\u00aa\u0001\u0000\u0000\u0000\u00b1!\u0001\u0000\u0000\u0000\u00b2"+
		"\u00b5\u0005\u0015\u0000\u0000\u00b3\u00b5\u0003$\u0012\u0000\u00b4\u00b2"+
		"\u0001\u0000\u0000\u0000\u00b4\u00b3\u0001\u0000\u0000\u0000\u00b5#\u0001"+
		"\u0000\u0000\u0000\u00b6\u00b7\u0006\u0012\uffff\uffff\u0000\u00b7\u00b8"+
		"\u0003\u0016\u000b\u0000\u00b8\u00be\u0001\u0000\u0000\u0000\u00b9\u00ba"+
		"\n\u0001\u0000\u0000\u00ba\u00bb\u0005\u0003\u0000\u0000\u00bb\u00bd\u0003"+
		"\u0016\u000b\u0000\u00bc\u00b9\u0001\u0000\u0000\u0000\u00bd\u00c0\u0001"+
		"\u0000\u0000\u0000\u00be\u00bc\u0001\u0000\u0000\u0000\u00be\u00bf\u0001"+
		"\u0000\u0000\u0000\u00bf%\u0001\u0000\u0000\u0000\u00c0\u00be\u0001\u0000"+
		"\u0000\u0000\u00c1\u00d1\u0003(\u0014\u0000\u00c2\u00c3\u0003(\u0014\u0000"+
		"\u00c3\u00c4\u0005\u0011\u0000\u0000\u00c4\u00c5\u0003\"\u0011\u0000\u00c5"+
		"\u00c6\u0005\u0012\u0000\u0000\u00c6\u00d1\u0001\u0000\u0000\u0000\u00c7"+
		"\u00c8\u0003(\u0014\u0000\u00c8\u00c9\u0003\u0014\n\u0000\u00c9\u00d1"+
		"\u0001\u0000\u0000\u0000\u00ca\u00cb\u0003(\u0014\u0000\u00cb\u00cc\u0003"+
		"\u0014\n\u0000\u00cc\u00cd\u0005\u0011\u0000\u0000\u00cd\u00ce\u0003\""+
		"\u0011\u0000\u00ce\u00cf\u0005\u0012\u0000\u0000\u00cf\u00d1\u0001\u0000"+
		"\u0000\u0000\u00d0\u00c1\u0001\u0000\u0000\u0000\u00d0\u00c2\u0001\u0000"+
		"\u0000\u0000\u00d0\u00c7\u0001\u0000\u0000\u0000\u00d0\u00ca\u0001\u0000"+
		"\u0000\u0000\u00d1\'\u0001\u0000\u0000\u0000\u00d2\u00d6\u0005\u001b\u0000"+
		"\u0000\u00d3\u00d5\u0007\u0005\u0000\u0000\u00d4\u00d3\u0001\u0000\u0000"+
		"\u0000\u00d5\u00d8\u0001\u0000\u0000\u0000\u00d6\u00d4\u0001\u0000\u0000"+
		"\u0000\u00d6\u00d7\u0001\u0000\u0000\u0000\u00d7)\u0001\u0000\u0000\u0000"+
		"\u00d8\u00d6\u0001\u0000\u0000\u0000\u00d9\u00db\u0005\u001c\u0000\u0000"+
		"\u00da\u00d9\u0001\u0000\u0000\u0000\u00db\u00dc\u0001\u0000\u0000\u0000"+
		"\u00dc\u00da\u0001\u0000\u0000\u0000\u00dc\u00dd\u0001\u0000\u0000\u0000"+
		"\u00dd+\u0001\u0000\u0000\u0000\r<IOYir\u007f\u00b0\u00b4\u00be\u00d0"+
		"\u00d6\u00dc";
	public static final ATN _ATN =
		new ATNDeserializer().deserialize(_serializedATN.toCharArray());
	static {
		_decisionToDFA = new DFA[_ATN.getNumberOfDecisions()];
		for (int i = 0; i < _ATN.getNumberOfDecisions(); i++) {
			_decisionToDFA[i] = new DFA(_ATN.getDecisionState(i), i);
		}
	}
}