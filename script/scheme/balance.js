!function(t,e){"object"==typeof exports&&"undefined"!=typeof module?module.exports=e():"function"==typeof define&&define.amd?define(e):(t=t||self).Nlvi=e()}(this,function(){"use strict";function o(t,e){if(!(t instanceof e))throw new TypeError("Cannot call a class as a function")}function r(t,e){for(var n=0;n<e.length;n++){var r=e[n];r.enumerable=r.enumerable||!1,r.configurable=!0,"value"in r&&(r.writable=!0),Object.defineProperty(t,r.key,r)}}function e(t,e,n){return e&&r(t.prototype,e),n&&r(t,n),t}function i(t){return(i=Object.setPrototypeOf?Object.getPrototypeOf:function(t){return t.__proto__||Object.getPrototypeOf(t)})(t)}function s(t,e){return(s=Object.setPrototypeOf||function(t,e){return t.__proto__=e,t})(t,e)}function u(t,e){return!e||"object"!=typeof e&&"function"!=typeof e?function(t){if(void 0===t)throw new ReferenceError("this hasn't been initialised - super() hasn't been called");return t}(t):e}function c(t,e,n){return(c="undefined"!=typeof Reflect&&Reflect.get?Reflect.get:function(t,e,n){var r=function(t,e){for(;!Object.prototype.hasOwnProperty.call(t,e)&&null!==(t=i(t)););return t}(t,e);if(r){var o=Object.getOwnPropertyDescriptor(r,e);return o.get?o.get.call(n):o.value}})(t,e,n||t)}function p(t,e){return function(t){if(Array.isArray(t))return t}(t)||function(t,e){var n=[],r=!0,o=!1,i=void 0;try{for(var s,u=t[Symbol.iterator]();!(r=(s=u.next()).done)&&(n.push(s.value),!e||n.length!==e);r=!0);}catch(t){o=!0,i=t}finally{try{r||null==u.return||u.return()}finally{if(o)throw i}}return n}(t,e)||function(){throw new TypeError("Invalid attempt to destructure non-iterable instance")}()}function a(t){return function(t){if(Array.isArray(t)){for(var e=0,n=new Array(t.length);e<t.length;e++)n[e]=t[e];return n}}(t)||function(t){if(Symbol.iterator in Object(t)||"[object Arguments]"===Object.prototype.toString.call(t))return Array.from(t)}(t)||function(){throw new TypeError("Invalid attempt to spread non-iterable instance")}()}var l=function(t,e){return(l=Object.setPrototypeOf||{__proto__:[]}instanceof Array&&function(t,e){t.__proto__=e}||function(t,e){for(var n in e)e.hasOwnProperty(n)&&(t[n]=e[n])})(t,e)};function h(t,e){function n(){this.constructor=t}l(t,e),t.prototype=null===e?Object.create(e):(n.prototype=e.prototype,new n)}function f(t){return"function"==typeof t}var n=!1,d={Promise:void 0,set useDeprecatedSynchronousErrorHandling(t){if(t);n=t},get useDeprecatedSynchronousErrorHandling(){return n}};function y(t){setTimeout(function(){throw t})}var b={closed:!0,next:function(t){},error:function(t){if(d.useDeprecatedSynchronousErrorHandling)throw t;y(t)},complete:function(){}},v=Array.isArray||function(t){return t&&"number"==typeof t.length};function m(t){return null!==t&&"object"==typeof t}function t(t){return Error.call(this),this.message=t?t.length+" errors occurred during unsubscription:\n"+t.map(function(t,e){return e+1+") "+t.toString()}).join("\n  "):"",this.name="UnsubscriptionError",this.errors=t,this}t.prototype=Object.create(Error.prototype);var w=t,g=function(){function o(t){this.closed=!1,this._parent=null,this._parents=null,this._subscriptions=null,t&&(this._unsubscribe=t)}var t;return o.prototype.unsubscribe=function(){var e,n=!1;if(!this.closed){var t=this._parent,r=this._parents,o=this._unsubscribe,i=this._subscriptions;this.closed=!0,this._parent=null,this._parents=null,this._subscriptions=null;for(var s=-1,u=r?r.length:0;t;)t.remove(this),t=++s<u&&r[s]||null;if(f(o))try{o.call(this)}catch(t){n=!0,e=t instanceof w?x(t.errors):[t]}if(v(i))for(s=-1,u=i.length;++s<u;){var c=i[s];if(m(c))try{c.unsubscribe()}catch(t){n=!0,e=e||[],t instanceof w?e=e.concat(x(t.errors)):e.push(t)}}if(n)throw new w(e)}},o.prototype.add=function(t){var e=t;switch(typeof t){case"function":e=new o(t);case"object":if(e===this||e.closed||"function"!=typeof e.unsubscribe)return e;if(this.closed)return e.unsubscribe(),e;if(!(e instanceof o)){var n=e;(e=new o)._subscriptions=[n]}break;default:if(!t)return o.EMPTY;throw new Error("unrecognized teardown "+t+" added to Subscription.")}if(e._addParent(this)){var r=this._subscriptions;r?r.push(e):this._subscriptions=[e]}return e},o.prototype.remove=function(t){var e=this._subscriptions;if(e){var n=e.indexOf(t);-1!==n&&e.splice(n,1)}},o.prototype._addParent=function(t){var e=this._parent,n=this._parents;return e!==t&&(e?n?-1===n.indexOf(t)&&(n.push(t),!0):(this._parents=[t],!0):(this._parent=t,!0))},o.EMPTY=((t=new o).closed=!0,t),o}();function x(t){return t.reduce(function(t,e){return t.concat(e instanceof w?e.errors:e)},[])}var _="function"==typeof Symbol?Symbol("rxSubscriber"):"@@rxSubscriber_"+Math.random(),S=function(o){function i(t,e,n){var r=o.call(this)||this;switch(r.syncErrorValue=null,r.syncErrorThrown=!1,r.syncErrorThrowable=!1,r.isStopped=!1,arguments.length){case 0:r.destination=b;break;case 1:if(!t){r.destination=b;break}if("object"==typeof t){t instanceof i?(r.syncErrorThrowable=t.syncErrorThrowable,(r.destination=t).add(r)):(r.syncErrorThrowable=!0,r.destination=new E(r,t));break}default:r.syncErrorThrowable=!0,r.destination=new E(r,t,e,n)}return r}return h(i,o),i.prototype[_]=function(){return this},i.create=function(t,e,n){var r=new i(t,e,n);return r.syncErrorThrowable=!1,r},i.prototype.next=function(t){this.isStopped||this._next(t)},i.prototype.error=function(t){this.isStopped||(this.isStopped=!0,this._error(t))},i.prototype.complete=function(){this.isStopped||(this.isStopped=!0,this._complete())},i.prototype.unsubscribe=function(){this.closed||(this.isStopped=!0,o.prototype.unsubscribe.call(this))},i.prototype._next=function(t){this.destination.next(t)},i.prototype._error=function(t){this.destination.error(t),this.unsubscribe()},i.prototype._complete=function(){this.destination.complete(),this.unsubscribe()},i.prototype._unsubscribeAndRecycle=function(){var t=this._parent,e=this._parents;return this._parent=null,this._parents=null,this.unsubscribe(),this.closed=!1,this.isStopped=!1,this._parent=t,this._parents=e,this},i}(g),E=function(u){function t(t,e,n,r){var o,i=u.call(this)||this;i._parentSubscriber=t;var s=i;return f(e)?o=e:e&&(o=e.next,n=e.error,r=e.complete,e!==b&&(f((s=Object.create(e)).unsubscribe)&&i.add(s.unsubscribe.bind(s)),s.unsubscribe=i.unsubscribe.bind(i))),i._context=s,i._next=o,i._error=n,i._complete=r,i}return h(t,u),t.prototype.next=function(t){if(!this.isStopped&&this._next){var e=this._parentSubscriber;d.useDeprecatedSynchronousErrorHandling&&e.syncErrorThrowable?this.__tryOrSetError(e,this._next,t)&&this.unsubscribe():this.__tryOrUnsub(this._next,t)}},t.prototype.error=function(t){if(!this.isStopped){var e=this._parentSubscriber,n=d.useDeprecatedSynchronousErrorHandling;if(this._error)n&&e.syncErrorThrowable?this.__tryOrSetError(e,this._error,t):this.__tryOrUnsub(this._error,t),this.unsubscribe();else if(e.syncErrorThrowable)n?(e.syncErrorValue=t,e.syncErrorThrown=!0):y(t),this.unsubscribe();else{if(this.unsubscribe(),n)throw t;y(t)}}},t.prototype.complete=function(){var t=this;if(!this.isStopped){var e=this._parentSubscriber;if(this._complete){var n=function(){return t._complete.call(t._context)};d.useDeprecatedSynchronousErrorHandling&&e.syncErrorThrowable?this.__tryOrSetError(e,n):this.__tryOrUnsub(n),this.unsubscribe()}else this.unsubscribe()}},t.prototype.__tryOrUnsub=function(t,e){try{t.call(this._context,e)}catch(t){if(this.unsubscribe(),d.useDeprecatedSynchronousErrorHandling)throw t;y(t)}},t.prototype.__tryOrSetError=function(e,t,n){if(!d.useDeprecatedSynchronousErrorHandling)throw new Error("bad call");try{t.call(this._context,n)}catch(t){return d.useDeprecatedSynchronousErrorHandling?(e.syncErrorValue=t,e.syncErrorThrown=!0):(y(t),!0)}return!1},t.prototype._unsubscribe=function(){var t=this._parentSubscriber;this._context=null,this._parentSubscriber=null,t.unsubscribe()},t}(S);var T="function"==typeof Symbol&&Symbol.observable||"@@observable";function k(){}var O=function(){function n(t){this._isScalar=!1,t&&(this._subscribe=t)}return n.prototype.lift=function(t){var e=new n;return e.source=this,e.operator=t,e},n.prototype.subscribe=function(t,e,n){var r=this.operator,o=function(t,e,n){if(t){if(t instanceof S)return t;if(t[_])return t[_]()}return t||e||n?new S(t,e,n):new S(b)}(t,e,n);if(o.add(r?r.call(o,this.source):this.source||d.useDeprecatedSynchronousErrorHandling&&!o.syncErrorThrowable?this._subscribe(o):this._trySubscribe(o)),d.useDeprecatedSynchronousErrorHandling&&o.syncErrorThrowable&&(o.syncErrorThrowable=!1,o.syncErrorThrown))throw o.syncErrorValue;return o},n.prototype._trySubscribe=function(e){try{return this._subscribe(e)}catch(t){d.useDeprecatedSynchronousErrorHandling&&(e.syncErrorThrown=!0,e.syncErrorValue=t),!function(t){for(;t;){var e=t.destination;if(t.closed||t.isStopped)return!1;t=e&&e instanceof S?e:null}return!0}(e)?console.warn(t):e.error(t)}},n.prototype.forEach=function(r,t){var o=this;return new(t=j(t))(function(t,e){var n;n=o.subscribe(function(t){try{r(t)}catch(t){e(t),n&&n.unsubscribe()}},e,t)})},n.prototype._subscribe=function(t){var e=this.source;return e&&e.subscribe(t)},n.prototype[T]=function(){return this},n.prototype.pipe=function(){for(var t=[],e=0;e<arguments.length;e++)t[e]=arguments[e];return 0===t.length?this:function(e){return e?1===e.length?e[0]:function(t){return e.reduce(function(t,e){return e(t)},t)}:k}(t)(this)},n.prototype.toPromise=function(t){var r=this;return new(t=j(t))(function(t,e){var n;r.subscribe(function(t){return n=t},function(t){return e(t)},function(){return t(n)})})},n.create=function(t){return new n(t)},n}();function j(t){if(t||(t=Promise),!t)throw new Error("no Promise impl found");return t}var A=function(r){function t(t,e){var n=r.call(this,t,e)||this;return n.scheduler=t,n.work=e,n.pending=!1,n}return h(t,r),t.prototype.schedule=function(t,e){if(void 0===e&&(e=0),this.closed)return this;this.state=t;var n=this.id,r=this.scheduler;return null!=n&&(this.id=this.recycleAsyncId(r,n,e)),this.pending=!0,this.delay=e,this.id=this.id||this.requestAsyncId(r,this.id,e),this},t.prototype.requestAsyncId=function(t,e,n){return void 0===n&&(n=0),setInterval(t.flush.bind(t,this),n)},t.prototype.recycleAsyncId=function(t,e,n){if(void 0===n&&(n=0),null!==n&&this.delay===n&&!1===this.pending)return e;clearInterval(e)},t.prototype.execute=function(t,e){if(this.closed)return new Error("executing a cancelled action");this.pending=!1;var n=this._execute(t,e);if(n)return n;!1===this.pending&&null!=this.id&&(this.id=this.recycleAsyncId(this.scheduler,this.id,null))},t.prototype._execute=function(t,e){var n=!1,r=void 0;try{this.work(t)}catch(t){n=!0,r=!!t&&t||new Error(t)}if(n)return this.unsubscribe(),r},t.prototype._unsubscribe=function(){var t=this.id,e=this.scheduler,n=e.actions,r=n.indexOf(this);this.work=null,this.state=null,this.pending=!1,this.scheduler=null,-1!==r&&n.splice(r,1),null!=t&&(this.id=this.recycleAsyncId(e,t,null)),this.delay=null},t}(function(n){function t(t,e){return n.call(this)||this}return h(t,n),t.prototype.schedule=function(t,e){return void 0===e&&(e=0),this},t}(g)),C=function(){function n(t,e){void 0===e&&(e=n.now),this.SchedulerAction=t,this.now=e}return n.prototype.schedule=function(t,e,n){return void 0===e&&(e=0),new this.SchedulerAction(this,t).schedule(n,e)},n.now=function(){return Date.now()},n}(),I=function(r){return function(t){for(var e=0,n=r.length;e<n&&!t.closed;e++)t.next(r[e]);t.closed||t.complete()}};function P(r,o){return new O(o?function(t){var e=new g,n=0;return e.add(o.schedule(function(){n!==r.length?(t.next(r[n++]),t.closed||e.add(this.schedule())):t.complete()})),e}:I(r))}var R=new(function(r){function o(t,e){void 0===e&&(e=C.now);var n=r.call(this,t,function(){return o.delegate&&o.delegate!==n?o.delegate.now():e()})||this;return n.actions=[],n.active=!1,n.scheduled=void 0,n}return h(o,r),o.prototype.schedule=function(t,e,n){return void 0===e&&(e=0),o.delegate&&o.delegate!==this?o.delegate.schedule(t,e,n):r.prototype.schedule.call(this,t,e,n)},o.prototype.flush=function(t){var e=this.actions;if(this.active)e.push(t);else{var n;this.active=!0;do{if(n=t.execute(t.state,t.delay))break}while(t=e.shift());if(this.active=!1,n){for(;t=e.shift();)t.unsubscribe();throw n}}},o}(C))(A);function H(e,n){return function(t){if("function"!=typeof e)throw new TypeError("argument is not a function. Are you looking for `mapTo()`?");return t.lift(new D(e,n))}}var D=function(){function t(t,e){this.project=t,this.thisArg=e}return t.prototype.call=function(t,e){return e.subscribe(new q(t,this.project,this.thisArg))},t}(),q=function(o){function t(t,e,n){var r=o.call(this,t)||this;return r.project=e,r.count=0,r.thisArg=n||r,r}return h(t,o),t.prototype._next=function(t){var e;try{e=this.project.call(this.thisArg,t,this.count++)}catch(t){return void this.destination.error(t)}this.destination.next(e)},t}(S),L=function(t){function e(){return null!==t&&t.apply(this,arguments)||this}return h(e,t),e.prototype.notifyNext=function(t,e,n,r,o){this.destination.next(e)},e.prototype.notifyError=function(t,e){this.destination.error(t)},e.prototype.notifyComplete=function(t){this.destination.complete()},e}(S),M=function(o){function t(t,e,n){var r=o.call(this)||this;return r.parent=t,r.outerValue=e,r.outerIndex=n,r.index=0,r}return h(t,o),t.prototype._next=function(t){this.parent.notifyNext(this.outerValue,t,this.outerIndex,this.index++,this)},t.prototype._error=function(t){this.parent.notifyError(t,this),this.unsubscribe()},t.prototype._complete=function(){this.parent.notifyComplete(this),this.unsubscribe()},t}(S),V=function(t){return function(e){return t.then(function(t){e.closed||(e.next(t),e.complete())},function(t){return e.error(t)}).then(null,y),e}};function N(){return"function"==typeof Symbol&&Symbol.iterator?Symbol.iterator:"@@iterator"}var X=N(),U=function(r){return function(t){for(var e=r[X]();;){var n=e.next();if(n.done){t.complete();break}if(t.next(n.value),t.closed)break}return"function"==typeof e.return&&t.add(function(){e.return&&e.return()}),t}},z=function(n){return function(t){var e=n[T]();if("function"!=typeof e.subscribe)throw new TypeError("Provided object does not correctly implement Symbol.observable");return e.subscribe(t)}},B=function(t){return t&&"number"==typeof t.length&&"function"!=typeof t};function W(t){return!!t&&"function"!=typeof t.subscribe&&"function"==typeof t.then}var F=function(e){if(e instanceof O)return function(t){return e._isScalar?(t.next(e.value),void t.complete()):e.subscribe(t)};if(e&&"function"==typeof e[T])return z(e);if(B(e))return I(e);if(W(e))return V(e);if(e&&"function"==typeof e[X])return U(e);var t=m(e)?"an invalid object":"'"+e+"'";throw new TypeError("You provided "+t+" where a stream was expected. You can provide an Observable, Promise, Array, or Iterable.")};function G(t,e,n,r,o){if(void 0===o&&(o=new M(t,n,r)),!o.closed)return F(e)(o)}function J(t,e){if(!e)return t instanceof O?t:new O(F(t));if(null!=t){if(function(t){return t&&"function"==typeof t[T]}(t))return function(r,o){return new O(o?function(e){var n=new g;return n.add(o.schedule(function(){var t=r[T]();n.add(t.subscribe({next:function(t){n.add(o.schedule(function(){return e.next(t)}))},error:function(t){n.add(o.schedule(function(){return e.error(t)}))},complete:function(){n.add(o.schedule(function(){return e.complete()}))}}))})),n}:z(r))}(t,e);if(W(t))return function(t,r){return new O(r?function(e){var n=new g;return n.add(r.schedule(function(){return t.then(function(t){n.add(r.schedule(function(){e.next(t),n.add(r.schedule(function(){return e.complete()}))}))},function(t){n.add(r.schedule(function(){return e.error(t)}))})})),n}:V(t))}(t,e);if(B(t))return P(t,e);if(function(t){return t&&"function"==typeof t[X]}(t)||"string"==typeof t)return function(e,n){if(!e)throw new Error("Iterable cannot be null");return new O(n?function(r){var o,t=new g;return t.add(function(){o&&"function"==typeof o.return&&o.return()}),t.add(n.schedule(function(){o=e[X](),t.add(n.schedule(function(){if(!r.closed){var t,e;try{var n=o.next();t=n.value,e=n.done}catch(t){return void r.error(t)}e?r.complete():(r.next(t),this.schedule())}}))})),t}:U(e))}(t,e)}throw new TypeError((null!==t&&typeof t||t)+" is not observable")}function Y(t,n,r,e){return f(r)&&(e=r,r=void 0),e?Y(t,n,r).pipe(H(function(t){return v(t)?e.apply(void 0,t):e(t)})):new O(function(e){!function t(e,n,r,o,i){var s;if(d=e,d&&"function"==typeof d.addEventListener&&"function"==typeof d.removeEventListener){var u=e;e.addEventListener(n,r,i),s=function(){return u.removeEventListener(n,r,i)}}else if(f=e,f&&"function"==typeof f.on&&"function"==typeof f.off){var c=e;e.on(n,r),s=function(){return c.off(n,r)}}else if(p=e,p&&"function"==typeof p.addListener&&"function"==typeof p.removeListener){var a=e;e.addListener(n,r),s=function(){return a.removeListener(n,r)}}else{if(!e||!e.length)throw new TypeError("Invalid event target");for(var l=0,h=e.length;l<h;l++)t(e[l],n,r,o,i)}var p;var f;var d;o.add(s)}(t,n,function(t){e.next(1<arguments.length?Array.prototype.slice.call(arguments):t)},e,r)})}function K(){for(var t=[],e=0;e<arguments.length;e++)t[e]=arguments[e];var n=t[t.length-1];return"function"==typeof n&&t.pop(),P(t,void 0).lift(new Q(n))}var Q=function(){function t(t){this.resultSelector=t}return t.prototype.call=function(t,e){return e.subscribe(new Z(t,this.resultSelector))},t}(),Z=function(o){function t(t,e,n){void 0===n&&(n=Object.create(null));var r=o.call(this,t)||this;return r.iterators=[],r.active=0,r.resultSelector="function"==typeof e?e:null,r.values=n,r}return h(t,o),t.prototype._next=function(t){var e=this.iterators;v(t)?e.push(new et(t)):e.push("function"==typeof t[X]?new tt(t[X]()):new nt(this.destination,this,t))},t.prototype._complete=function(){var t=this.iterators,e=t.length;if(this.unsubscribe(),0!==e){this.active=e;for(var n=0;n<e;n++){var r=t[n];if(r.stillUnsubscribed)this.destination.add(r.subscribe(r,n));else this.active--}}else this.destination.complete()},t.prototype.notifyInactive=function(){this.active--,0===this.active&&this.destination.complete()},t.prototype.checkIterators=function(){for(var t=this.iterators,e=t.length,n=this.destination,r=0;r<e;r++){if("function"==typeof(s=t[r]).hasValue&&!s.hasValue())return}var o=!1,i=[];for(r=0;r<e;r++){var s,u=(s=t[r]).next();if(s.hasCompleted()&&(o=!0),u.done)return void n.complete();i.push(u.value)}this.resultSelector?this._tryresultSelector(i):n.next(i),o&&n.complete()},t.prototype._tryresultSelector=function(t){var e;try{e=this.resultSelector.apply(this,t)}catch(t){return void this.destination.error(t)}this.destination.next(e)},t}(S),tt=function(){function t(t){this.iterator=t,this.nextResult=t.next()}return t.prototype.hasValue=function(){return!0},t.prototype.next=function(){var t=this.nextResult;return this.nextResult=this.iterator.next(),t},t.prototype.hasCompleted=function(){var t=this.nextResult;return t&&t.done},t}(),et=function(){function t(t){this.array=t,this.index=0,this.length=0,this.length=t.length}return t.prototype[X]=function(){return this},t.prototype.next=function(t){var e=this.index++;return e<this.length?{value:this.array[e],done:!1}:{value:null,done:!0}},t.prototype.hasValue=function(){return this.index<this.array.length},t.prototype.hasCompleted=function(){return this.array.length===this.index},t}(),nt=function(o){function t(t,e,n){var r=o.call(this,t)||this;return r.parent=e,r.observable=n,r.stillUnsubscribed=!0,r.buffer=[],r.isComplete=!1,r}return h(t,o),t.prototype[X]=function(){return this},t.prototype.next=function(){var t=this.buffer;return 0===t.length&&this.isComplete?{value:null,done:!0}:{value:t.shift(),done:!1}},t.prototype.hasValue=function(){return 0<this.buffer.length},t.prototype.hasCompleted=function(){return 0===this.buffer.length&&this.isComplete},t.prototype.notifyComplete=function(){0<this.buffer.length?(this.isComplete=!0,this.parent.notifyInactive()):this.destination.complete()},t.prototype.notifyNext=function(t,e,n,r,o){this.buffer.push(e),this.parent.checkIterators()},t.prototype.subscribe=function(t,e){return G(this,this.observable,this,e)},t}(L);var rt=function(){function t(t,e){this.dueTime=t,this.scheduler=e}return t.prototype.call=function(t,e){return e.subscribe(new ot(t,this.dueTime,this.scheduler))},t}(),ot=function(o){function t(t,e,n){var r=o.call(this,t)||this;return r.dueTime=e,r.scheduler=n,r.debouncedSubscription=null,r.lastValue=null,r.hasValue=!1,r}return h(t,o),t.prototype._next=function(t){this.clearDebounce(),this.lastValue=t,this.hasValue=!0,this.add(this.debouncedSubscription=this.scheduler.schedule(it,this.dueTime,this))},t.prototype._complete=function(){this.debouncedNext(),this.destination.complete()},t.prototype.debouncedNext=function(){if(this.clearDebounce(),this.hasValue){var t=this.lastValue;this.lastValue=null,this.hasValue=!1,this.destination.next(t)}},t.prototype.clearDebounce=function(){var t=this.debouncedSubscription;null!==t&&(this.remove(t),t.unsubscribe(),this.debouncedSubscription=null)},t}(S);function it(t){t.debouncedNext()}function st(e,n){return function(t){return t.lift(new ut(e,n))}}var ut=function(){function t(t,e){this.predicate=t,this.thisArg=e}return t.prototype.call=function(t,e){return e.subscribe(new ct(t,this.predicate,this.thisArg))},t}(),ct=function(o){function t(t,e,n){var r=o.call(this,t)||this;return r.predicate=e,r.thisArg=n,r.count=0,r}return h(t,o),t.prototype._next=function(t){var e;try{e=this.predicate.call(this.thisArg,t,this.count++)}catch(t){return void this.destination.error(t)}e&&this.destination.next(t)},t}(S);function at(e,o){return"function"==typeof o?function(t){return t.pipe(at(function(n,r){return J(e(n,r)).pipe(H(function(t,e){return o(n,t,r,e)}))}))}:function(t){return t.lift(new lt(e))}}var lt=function(){function t(t){this.project=t}return t.prototype.call=function(t,e){return e.subscribe(new ht(t,this.project))},t}(),ht=function(r){function t(t,e){var n=r.call(this,t)||this;return n.project=e,n.index=0,n}return h(t,r),t.prototype._next=function(t){var e,n=this.index++;try{e=this.project(t,n)}catch(t){return void this.destination.error(t)}this._innerSub(e,t,n)},t.prototype._innerSub=function(t,e,n){var r=this.innerSubscription;r&&r.unsubscribe();var o=new M(this,void 0,void 0);this.destination.add(o),this.innerSubscription=G(this,t,e,n,o)},t.prototype._complete=function(){var t=this.innerSubscription;t&&!t.closed||r.prototype._complete.call(this),this.unsubscribe()},t.prototype._unsubscribe=function(){this.innerSubscription=null},t.prototype.notifyComplete=function(t){this.destination.remove(t),this.innerSubscription=null,this.isStopped&&r.prototype._complete.call(this)},t.prototype.notifyNext=function(t,e,n,r,o){this.destination.next(e)},t}(L);var pt=function(){function t(t,e){this.observables=t,this.project=e}return t.prototype.call=function(t,e){return e.subscribe(new ft(t,this.observables,this.project))},t}(),ft=function(u){function t(t,e,n){var r=u.call(this,t)||this;r.observables=e,r.project=n,r.toRespond=[];var o=e.length;r.values=new Array(o);for(var i=0;i<o;i++)r.toRespond.push(i);for(i=0;i<o;i++){var s=e[i];r.add(G(r,s,s,i))}return r}return h(t,u),t.prototype.notifyNext=function(t,e,n,r,o){this.values[n]=e;var i=this.toRespond;if(0<i.length){var s=i.indexOf(n);-1!==s&&i.splice(s,1)}},t.prototype.notifyComplete=function(){},t.prototype._next=function(t){if(0===this.toRespond.length){var e=[t].concat(this.values);this.project?this._tryProject(e):this.destination.next(e)}},t.prototype._tryProject=function(t){var e;try{e=this.project.apply(this,t)}catch(t){return void this.destination.error(t)}this.destination.next(e)},t}(L),dt="undefined"!=typeof window&&window,yt="undefined"!=typeof self&&"undefined"!=typeof WorkerGlobalScope&&self instanceof WorkerGlobalScope&&self,bt="undefined"!=typeof global&&global,vt=dt||bt||yt;function mt(t,e){return void 0===e&&(e=null),new Tt({method:"GET",url:t,headers:e})}function wt(t,e,n){return new Tt({method:"POST",url:t,body:e,headers:n})}function gt(t,e){return new Tt({method:"DELETE",url:t,headers:e})}function xt(t,e,n){return new Tt({method:"PUT",url:t,body:e,headers:n})}function _t(t,e,n){return new Tt({method:"PATCH",url:t,body:e,headers:n})}var St=H(function(t,e){return t.response});function Et(t,e){return St(new Tt({method:"GET",url:t,responseType:"json",headers:e}))}var Tt=function(o){function e(t){var e=o.call(this)||this,n={async:!0,createXHR:function(){return this.crossDomain?function(){if(vt.XMLHttpRequest)return new vt.XMLHttpRequest;if(vt.XDomainRequest)return new vt.XDomainRequest;throw new Error("CORS is not supported by your browser")}():function(){if(vt.XMLHttpRequest)return new vt.XMLHttpRequest;var t=void 0;try{for(var e=["Msxml2.XMLHTTP","Microsoft.XMLHTTP","Msxml2.XMLHTTP.4.0"],n=0;n<3;n++)try{if(new vt.ActiveXObject(t=e[n]))break}catch(t){}return new vt.ActiveXObject(t)}catch(t){throw new Error("XMLHttpRequest is not supported by your browser")}}()},crossDomain:!0,withCredentials:!1,headers:{},method:"GET",responseType:"json",timeout:0};if("string"==typeof t)n.url=t;else for(var r in t)t.hasOwnProperty(r)&&(n[r]=t[r]);return e.request=n,e}var t;return h(e,o),e.prototype._subscribe=function(t){return new kt(t,this.request)},e.create=((t=function(t){return new e(t)}).get=mt,t.post=wt,t.delete=gt,t.put=xt,t.patch=_t,t.getJSON=Et,t),e}(O),kt=function(o){function t(t,e){var n=o.call(this,t)||this;n.request=e,n.done=!1;var r=e.headers=e.headers||{};return e.crossDomain||n.getHeader(r,"X-Requested-With")||(r["X-Requested-With"]="XMLHttpRequest"),n.getHeader(r,"Content-Type")||vt.FormData&&e.body instanceof vt.FormData||void 0===e.body||(r["Content-Type"]="application/x-www-form-urlencoded; charset=UTF-8"),e.body=n.serializeBody(e.body,n.getHeader(e.headers,"Content-Type")),n.send(),n}return h(t,o),t.prototype.next=function(t){this.done=!0;var e,n=this.xhr,r=this.request,o=this.destination;try{e=new Ot(t,n,r)}catch(t){return o.error(t)}o.next(e)},t.prototype.send=function(){var t=this.request,e=this.request,n=e.user,r=e.method,o=e.url,i=e.async,s=e.password,u=e.headers,c=e.body;try{var a=this.xhr=t.createXHR();this.setupEvents(a,t),n?a.open(r,o,i,n,s):a.open(r,o,i),i&&(a.timeout=t.timeout,a.responseType=t.responseType),"withCredentials"in a&&(a.withCredentials=!!t.withCredentials),this.setHeaders(a,u),c?a.send(c):a.send()}catch(t){this.error(t)}},t.prototype.serializeBody=function(e,t){if(!e||"string"==typeof e)return e;if(vt.FormData&&e instanceof vt.FormData)return e;if(t){var n=t.indexOf(";");-1!==n&&(t=t.substring(0,n))}switch(t){case"application/x-www-form-urlencoded":return Object.keys(e).map(function(t){return encodeURIComponent(t)+"="+encodeURIComponent(e[t])}).join("&");case"application/json":return JSON.stringify(e);default:return e}},t.prototype.setHeaders=function(t,e){for(var n in e)e.hasOwnProperty(n)&&t.setRequestHeader(n,e[n])},t.prototype.getHeader=function(t,e){for(var n in t)if(n.toLowerCase()===e.toLowerCase())return t[n]},t.prototype.setupEvents=function(t,e){var n=e.progressSubscriber;function i(t){var e,n=i.subscriber,r=i.progressSubscriber,o=i.request;r&&r.error(t);try{e=new Ct(this,o)}catch(t){e=t}n.error(e)}if((t.ontimeout=i).request=e,i.subscriber=this,i.progressSubscriber=n,t.upload&&"withCredentials"in t){var r,s;if(n)r=function(t){r.progressSubscriber.next(t)},vt.XDomainRequest?t.onprogress=r:t.upload.onprogress=r,r.progressSubscriber=n;(t.onerror=s=function(t){var e,n=s.progressSubscriber,r=s.subscriber,o=s.request;n&&n.error(t);try{e=new $t("ajax error",this,o)}catch(t){e=t}r.error(e)}).request=e,s.subscriber=this,s.progressSubscriber=n}function o(t){}function u(t){var e=u.subscriber,n=u.progressSubscriber,r=u.request;if(4===this.readyState){var o=1223===this.status?204:this.status;if(0===o&&(o=("text"===this.responseType?this.response||this.responseText:this.response)?200:0),o<400)n&&n.complete(),e.next(t),e.complete();else{n&&n.error(t);var i=void 0;try{i=new $t("ajax error "+o,this,r)}catch(t){i=t}e.error(i)}}}(t.onreadystatechange=o).subscriber=this,o.progressSubscriber=n,o.request=e,(t.onload=u).subscriber=this,u.progressSubscriber=n,u.request=e},t.prototype.unsubscribe=function(){var t=this.xhr;!this.done&&t&&4!==t.readyState&&"function"==typeof t.abort&&t.abort(),o.prototype.unsubscribe.call(this)},t}(S),Ot=function(){return function(t,e,n){this.originalEvent=t,this.xhr=e,this.request=n,this.status=e.status,this.responseType=e.responseType||n.responseType,this.response=At(this.responseType,e)}}();function jt(t,e,n){return Error.call(this),this.message=t,this.name="AjaxError",this.xhr=e,this.request=n,this.status=e.status,this.responseType=e.responseType||n.responseType,this.response=At(this.responseType,e),this}jt.prototype=Object.create(Error.prototype);var $t=jt;function At(t,e){switch(t){case"json":return function(t){return"response"in t?t.responseType?t.response:JSON.parse(t.response||t.responseText||"null"):JSON.parse(t.responseText||"null")}(e);case"xml":return e.responseXML;case"text":default:return"response"in e?e.response:e.responseText}}var Ct=function(t,e){return $t.call(this,"ajax timeout",t,e),this.name="AjaxTimeoutError",this},It=Tt.create;function Pt(t,e){var n=It({url:t,responseType:"xml"}).pipe(H(function(t){return t.response}),H(function(t){return t.querySelectorAll("entry")}),H(function(t){return a(t).map(function(t){return{title:t.getElementsByTagName("title")[0].textContent,url:t.getElementsByTagName("url")[0].textContent,content:t.getElementsByTagName("content")[0].textContent}})})),r=Y(document.getElementById(e),"input").pipe(function(e,n){return void 0===n&&(n=R),function(t){return t.lift(new rt(e,n))}}(500),H(function(t){return t.target.value.trim()}));return K(r.pipe(H(function(t){return!!t})),r.pipe(function(){for(var n=[],t=0;t<arguments.length;t++)n[t]=arguments[t];return function(t){var e;return"function"==typeof n[n.length-1]&&(e=n.pop()),t.lift(new pt(n,e))}}(n),H(function(t){var e=p(t,2),i=e[0];return e[1].filter(function(t){var e=t.content;return 0<=t.title.indexOf(i)||0<=e.indexOf(i)}).map(function(t){var e=new RegExp("(".concat(i,")"),"gi"),n=t.title.replace(e,'<strong class="search-keyword">$1</strong>'),r=t.content.replace(/<[^>]+>/g,""),o=r.indexOf(i);return function(o){for(var t=1;t<arguments.length;t++){var i=null!=arguments[t]?arguments[t]:{},e=Object.keys(i);"function"==typeof Object.getOwnPropertySymbols&&(e=e.concat(Object.getOwnPropertySymbols(i).filter(function(t){return Object.getOwnPropertyDescriptor(i,t).enumerable}))),e.forEach(function(t){var e,n,r;r=i[n=t],n in(e=o)?Object.defineProperty(e,n,{value:r,enumerable:!0,configurable:!0,writable:!0}):e[n]=r})}return o}({},t,{title:n,content:r=r.slice(o<20?0:o-20,o<0?100:o+80).replace(e,'<strong class="search-keyword">$1</strong>')})})}))).pipe(H(function(t){var e=p(t,2);return e[0]?e[1]:[]}))}var Rt=function(){function r(t){o(this,r),this.config=t,this.theme=t.theme,this.scrollArr=[]}return e(r,[{key:"init",value:function(){var e=this,n=r.utils,t={smoothScroll:function(){$(".toc-link").on("click",function(){$("html, body").animate({scrollTop:$($.attr(this,"href")).offset().top-200})})},picPos:function(){var e=this;$(".post-content").each(function(){$(this).find("img").each(function(){$(this).parent("p").css("text-align","center");var t='<img src="'.concat(this.src,'"');e.theme.lazy&&(t='<img data-src="'.concat(this.src,'" class="lazyload"')),$(this).replaceWith('<a href="'.concat(this.src,'" data-title="').concat(this.alt,'" data-lightbox="group">').concat(t,' alt="').concat(this.alt,'"></a>'))})})},showComments:function(){var t=this;$("#com-switch").on("click",function(){n("iss","#post-comments").display()?($("#post-comments").css("display","block").addClass("syuanpi fadeInDown"),$(t).removeClass("syuanpi").css("transform","rotate(180deg)")):($(t).addClass("syuanpi").css("transform",""),n("cls","#post-comments").opreate("fadeInDown","remove"),n("ani","#post-comments").end("fadeOutUp",function(){$(this).css("display","none")}))})}};return r.opScroll(this.scrollArr),Object.values(t).forEach(function(t){return t.call(e)})}},{key:"back2top",value:function(){$(".toTop").on("click",function(){$("html, body").animate({scrollTop:0})})}},{key:"pushHeader",value:function(){var e=this.utils("cls","#mobile-header");this.scrollArr.push(function(t){e.opreate("header-scroll",5<t?"add":"remove")})}},{key:"updateRound",value:function(t){var e=Math.floor(t/($(document).height()-$(window).height())*100);$("#scrollpercent").html(e)}},{key:"showToc",value:function(){var s=r.utils,u=$(".toc-link"),c=$(".headerlink");this.scrollArr.push(function(e){var t=$.map(c,function(t){return $(t).offset().top});$(".title-link a").each(function(){var t=s("cls",this);0<=e&&e<230?t.opreate("active"):t.opreate("active","remove")});for(var n=0;n<u.length;n++){var r=t[n],o=n+1===u.length?1/0:t[n+1],i=s("cls",u[n]);r<e+210&&e+210<=o?i.opreate("active"):i.opreate("active","remove")}})}},{key:"titleStatus",value:function(){var e,n=document.title;document.addEventListener("visibilitychange",function(){var t=Math.floor($(window).scrollTop()/($(document).height()-$(window).height())*100);$(document).height()-$(window).height()==0&&(function(t){throw new Error('"'+t+'" is read-only')}("sct"),t=100),document.hidden?(clearTimeout(e),document.title="Read "+t+"% · "+n):(document.title="Welcome Back · "+n,e=setTimeout(function(){document.title=n},3e3))})}},{key:"showReward",value:function(){if(this.theme.reward){var t=r.utils,e=t("ani","#reward-btn");$("#reward-btn").click(function(){t("iss","#reward-wrapper").display()?($("#reward-wrapper").css("display","flex"),e.end("clarity")):e.end("melt",function(){$("#reward-wrapper").hide()})})}}},{key:"listenExit",value:function(t,e){Y(t,"keydown").pipe(st(function(t){return 27===t.keyCode})).subscribe(function(){return e()})}},{key:"depth",value:function(t,e){var n=this.utils,r=n("cls","body"),o=n("cls",".container-inner");r.exist("under")?(r.opreate("under","remove"),o.opreate("under","remove"),e.call(this)):(r.opreate("under","add"),o.opreate("under","add"),t.call(this))}},{key:"tagcloud",value:function(){var e=this,t=this.utils,n=$("#tags"),r=t("cls","#tagcloud"),o=t("ani","#tagcloud"),i=t("cls","#search"),s=t("ani","#search"),u=function(){r.opreate("shuttleIn","remove"),o.end("zoomOut",function(){r.opreate("syuanpi show","remove")})},c=function(){e.depth(function(){return r.opreate("syuanpi shuttleIn show")},u)};this.listenExit(n,c),this.listenExit(document.getElementsByClassName("tagcloud-taglist"),c),n.on("click",function(){if(i.exist("show"))return r.opreate("syuanpi shuttleIn show"),i.opreate("shuttleIn","remove"),void s.end("zoomOut",function(){i.opreate("syuanpi show","remove")});c()}),$("#tagcloud").on("click",function(t){t.stopPropagation(),"DIV"===t.target.tagName&&e.depth(function(){return r.opreate("syuanpi shuttleIn show")},u)});var a=Y(document.querySelectorAll(".tagcloud-tag button"),"click").pipe(H(function(t){return t.target})),l=J(document.querySelectorAll(".tagcloud-postlist")),h=l.pipe(H(function(t){return t.classList.remove("active")}));K(a.pipe(at(function(){return h})),a).pipe(H(function(t){var e=p(t,2);return e[1]}),at(function(e){return l.pipe(st(function(t){return t.firstElementChild.innerHTML.trim()===e.innerHTML.trim()}))})).subscribe(function(t){return t.classList.add("active")})}},{key:"search",value:function(){var e=this;if(this.theme.search){$("body").append('<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"><\/script>');var t=this.utils,n=$("#search-btn"),r=$("#search-result"),o=t("cls","#search"),i=t("ani","#search"),s=t("cls","#tagcloud"),u=t("ani","#tagcloud"),c=function(){o.opreate("shuttleIn","remove"),i.end("zoomOut",function(){o.opreate("syuanpi show","remove")})},a=function(){e.depth(function(){return o.opreate("syuanpi shuttleIn show")},c)};this.listenExit(document.getElementById("search"),a),n.on("click",function(){if(s.exist("show"))return o.opreate("syuanpi shuttleIn show"),s.opreate("shuttleIn","remove"),void u.end("zoomOut",function(){s.opreate("syuanpi show","remove")});a()}),$("#search").on("click",function(t){t.stopPropagation(),"DIV"===t.target.tagName&&e.depth(function(){return o.opreate("syuanpi shuttleIn show")},c)}),Pt("".concat(this.config.baseUrl,"search.xml"),"search-input").subscribe(function(t){var e,n=t.map(function(t){var e=t.title,n=t.content;return'\n          <li class="search-result-item">\n            <a href="'.concat(t.url,'"><h2>').concat(e,"</h2></a>\n            <p>").concat(n,"</p>\n          </li>\n        ")});r.html((e=n.join(""),'<ul class="search-result-list syuanpi fadeInUpShort">'.concat(e,"</ul>")))})}}},{key:"headerMenu",value:function(){var t=this,e=this.utils,n=e("cls",".mobile-header-body"),r=e("cls",".header-menu-line"),o=$("#mobile-tags"),i=e("cls","#tagcloud");o.on("click",function(){n.opreate("show","remove"),r.opreate("show","remove"),i.opreate("syuanpi shuttleIn show")}),$("#mobile-left").on("click",function(){t.depth(function(){n.opreate("show"),r.opreate("show")},function(){n.opreate("show","remove"),r.opreate("show","remove")})})}},{key:"pjax",value:function(){if(this.theme.pjax){var t=this.utils,e=t("cls",".container-inner"),n=t("cls",".header"),r=t("cls",".header-wrapper");$(document).pjax(".container-inner a",".container-inner",{fragment:"container-inner"}),$(document).on("pjax:send",function(){e.opreate("syuanpi fadeOutLeftShort"),r.opreate("syuanpi fadeOutLeftShort"),n.opreate("melt")})}}},{key:"bootstarp",value:function(){this.showToc(),this.back2top(),this.switchToc(),this.titleStatus(),this.init(),this.pushHeader(),this.tagcloud(),this.search(),this.showReward(),this.headerMenu(),this.pjax()}}],[{key:"utils",value:function(t,e){var n=this,r=function(n){return{opreate:function(t,e){return"remove"===e?$(n).removeClass(t):$(n).addClass(t)},exist:function(t){return $(n).hasClass(t)}}};return{cls:r,iss:function(t){return{banderole:function(){return"banderole"===n.theme.scheme},balance:function(){return"balance"===n.theme.scheme},display:function(){return"none"===$(t).css("display")}}},ani:function(n){return{close:function(){return r.opreate(".syuanpi","syuanpi","remove")},end:function(t,e){$(n).addClass(t).one("webkitAnimationEnd AnimationEnd",function(){$(n).removeClass(t),e&&e.call(null,n)})}}}}[t](e)}},{key:"opScroll",value:function(t){var e=Y(window,"scroll").pipe(H(function(t){return t.target.scrollingElement.scrollTop}));t.length&&e.subscribe(function(e){return t.forEach(function(t){return t(e)})})}}]),r}();return function(t){function n(t){var e;return o(this,n),(e=u(this,i(n).call(this,t))).utils=Rt.utils,e}return function(t,e){if("function"!=typeof e&&null!==e)throw new TypeError("Super expression must either be null or a function");t.prototype=Object.create(e&&e.prototype,{constructor:{value:t,writable:!0,configurable:!0}}),e&&s(t,e)}(n,Rt),e(n,[{key:"back2top",value:function(){var e=this;this.utils("cls","#backtop").opreate("melt","remove"),this.scrollArr.push(function(t){e.updateRound(t)}),c(i(n.prototype),"back2top",this).call(this)}},{key:"switchToc",value:function(){var t=this.utils,e=t("cls","#header"),n=t("cls",".post-toc");$("#switch-toc").on("click",function(t){t.stopPropagation(),e.opreate("show_toc"),n.opreate("hide","remove")}),$("#header").on("click",function(){e.exist("show_toc")&&(e.opreate("show_toc","remove"),n.opreate("hide"))})}}]),n}()});
